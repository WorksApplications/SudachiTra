import logging
import sys
import itertools as it

import numpy as np
import tensorflow as tf
from transformers import (
    TFAutoModelForMultipleChoice,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


def setup_args(data_args, datasets):
    # set dataset column name, assuming to use convert_dataset.py
    dataset_key = list(datasets.keys())[0]  # at least one data file exists
    clm_names = datasets[dataset_key].column_names

    n_choices = 0
    while f"choice_{n_choices}" in clm_names:
        n_choices += 1

    data_args.context_column = "context"
    data_args.choice_columns = [f"choice_{i}" for i in range(n_choices)]
    data_args.label_column = "label"
    return data_args


def tokenize_texts(datadict, pretok, data_args):
    column_names = data_args.choice_columns + [data_args.context_column]

    def subfunc(examples):
        for c in column_names:
            examples[c] = [pretok(s) for s in examples[c]]
        return examples

    datadict = datadict.map(subfunc, batched=True)
    return datadict


def preprocess_dataset(dataset, data_args, tokenizer, max_length):
    context_column = data_args.context_column
    choice_columns = data_args.choice_columns
    label_column = data_args.label_column
    n_choices = len(choice_columns)

    dataset_key = list(dataset.keys())[0]
    data_columns = [
        c for c in dataset[dataset_key].column_names if c != label_column]

    def subfunc(examples):
        first_sentences = ([c] * n_choices for c in examples[context_column])
        first_sentences = list(it.chain(*first_sentences))
        second_sentences = (examples[clm] for clm in choice_columns)
        second_sentences = list(it.chain(*zip(*second_sentences)))

        tokenized = tokenizer(first_sentences, second_sentences,
                              max_length=max_length, truncation=True)

        # un-flatten
        data = {k: [v[i:i+n_choices] for i in range(0, len(v), n_choices)]
                for k, v in tokenized.items()}

        # keep label column as it is, assuming it contains 0-indexed integer
        return data

    dataset = dataset.map(subfunc, batched=True, remove_columns=data_columns)
    return dataset


def convert_dataset_for_tensorflow(
    dataset, batch_size, dataset_mode="variable_batch", shuffle=True, drop_remainder=False
):
    def densify_ragged_batch(features, label=None):
        features = {
            feature: ragged_tensor.to_tensor(shape=batch_shape[feature]) for feature, ragged_tensor in features.items()
        }
        if label is None:
            return features
        else:
            return features, label

    # convert all columns except "label".
    # dataset should not have unneccessary columns.
    feature_keys = list(set(dataset.features.keys()) - {"label"})

    # trim input length for each batch
    if dataset_mode == "variable_batch":
        batch_shape = {key: None for key in feature_keys}
        data = {key: tf.ragged.constant(dataset[key]) for key in feature_keys}
    elif dataset_mode == "constant_batch":
        data = {key: tf.ragged.constant(dataset[key]) for key in feature_keys}
        batch_shape = {
            key: tf.concat(
                ([batch_size], ragged_tensor.bounding_shape()[1:]), axis=0)
            for key, ragged_tensor in data.items()
        }
    else:
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}")

    if "label" in dataset.features:
        labels = tf.convert_to_tensor(np.array(dataset["label"]))
        tf_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    else:
        tf_dataset = tf.data.Dataset.from_tensor_slices(data)

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(dataset))

    # ref: https://github.com/tensorflow/tensorflow/issues/42146
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    tf_dataset = (
        tf_dataset.with_options(options)
        .batch(batch_size=batch_size, drop_remainder=drop_remainder)
        .map(densify_ragged_batch)
    )
    return tf_dataset


def setup_model(model_name_or_path, config, training_args, from_pt=False):
    model = TFAutoModelForMultipleChoice.from_pretrained(
        model_name_or_path,
        config=config,
        from_pt=from_pt,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=training_args.learning_rate,
        beta_1=training_args.adam_beta1,
        beta_2=training_args.adam_beta2,
        epsilon=training_args.adam_epsilon,
        clipnorm=training_args.max_grad_norm,
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model


def evaluate_model(model, processed_dataset, tf_dataset, output_dir=None, stage="eval"):
    metrics = model.evaluate(tf_dataset, return_dict=True)
    labels = processed_dataset["label"]

    if output_dir is not None:
        predictions = model.predict(tf_dataset)["logits"]
        predicted_class = np.argmax(predictions, axis=1)

        output_file = output_dir / f"{stage}_predictions.tsv"
        with open(output_file, "w") as writer:
            writer.write("index\tlabel\tprediction\n")
            for index, (label, item) in enumerate(zip(labels, predicted_class)):
                writer.write(f"{index}\t{label}\t{item}\n")

    return metrics
