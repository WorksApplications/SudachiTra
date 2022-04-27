import logging
import sys

import numpy as np
import tensorflow as tf
from transformers import (
    TFAutoModelForSequenceClassification,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger.setLevel(logging.INFO)


def setup_args(data_args, raw_datadict):
    # num_label to initialize model
    dataset_key = list(raw_datadict.keys())[0]  # at least one data file exists
    label_list = raw_datadict[dataset_key].unique("label")
    logger.info(f"classification task with {len(label_list)} labels.")

    data_args.label_list = sorted(label_list)
    data_args.label2id = {l: i for i, l in enumerate(label_list)}

    # columns of input text
    data_columns = [
        c for c in raw_datadict[dataset_key].column_names if c != "label"]
    if "sentence1" in data_columns:
        if "sentence2" in data_columns:
            text_columns = ["sentence1", "sentence2"]
        else:
            text_columns = ["sentence1"]
    else:
        text_columns = data_columns[:2]

    data_args.data_columns = data_columns
    data_args.text_columns = text_columns
    return data_args


def pretokenize_texts(raw_datadict, pretok, data_args):
    def subfunc(examples):
        for c in data_args.text_columns:
            examples[c] = [pretok(s) for s in examples[c]]
        return examples

    raw_datadict = raw_datadict.map(subfunc, batched=True)
    return raw_datadict


def preprocess_dataset(raw_datadict, data_args, tokenizer, max_length):
    # Truncate text before tokenization for sudachi, which has a input bytes limit.
    # This may affect the result with a large max_length (tokens).
    MAX_CHAR_LENGTH = 2**14

    def subfunc(examples):
        # Tokenize texts
        texts = ([s[:MAX_CHAR_LENGTH] for s in examples[c]]
                 for c in data_args.text_columns)
        result = tokenizer(*texts, max_length=max_length, truncation=True)

        # Map labels to ids
        if "label" in examples:
            result["label"] = [
                (data_args.label2id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datadict = raw_datadict.map(
        subfunc,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        remove_columns=data_args.data_columns
    )
    return datadict


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
    model = TFAutoModelForSequenceClassification.from_pretrained(
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
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model


def evaluate_model(model, dataset, tf_dataset, label2id, output_dir=None, stage="eval"):
    predictions = model.predict(tf_dataset)["logits"]
    predicted_class = np.argmax(predictions, axis=1)

    labels = dataset["label"]
    acc = sum(predicted_class == labels) / len(labels)
    metrics = {"accuracy": acc}

    if output_dir is not None:
        id2label = {i: l for l, i in label2id.items()}
        output_file = output_dir / f"{stage}_predictions.tsv"
        with open(output_file, "w") as writer:
            writer.write("index\tlabel\tprediction\n")
            for index, (label, item) in enumerate(zip(labels, predicted_class)):
                label = id2label[label]
                item = id2label[item]
                writer.write(f"{index}\t{label}\t{item}\n")

    return metrics
