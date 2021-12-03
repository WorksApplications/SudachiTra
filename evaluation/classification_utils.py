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
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


def preprocess_dataset(dataset, data_args, tokenizer, max_length):
    # select columns to tokenize
    dataset_name = list(dataset.keys())[0]
    data_columns = [
        c for c in dataset[dataset_name].column_names if c != "label"]
    if "sentence1" in data_columns:
        if "sentence2" in data_columns:
            column_names = ["sentence1", "sentence2"]
        else:
            column_names = ["sentence1"]
    else:
        column_names = data_columns[:2]

    def subfunc(examples):
        # Tokenize texts
        texts = (examples[c] for c in column_names)
        result = tokenizer(*texts, max_length=max_length, truncation=True)

        # Map labels to ids
        if "label" in examples:
            result["label"] = [
                (data_args.label2id[l] if l != -1 else -1) for l in examples["label"]]
        return result

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

    # trim input length for each batch
    feature_keys = list(set(dataset.features.keys()) - {"label"})
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
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    tf_dataset = tf_dataset.with_options(options)

    tf_dataset = tf_dataset.batch(
        batch_size=batch_size, drop_remainder=drop_remainder).map(densify_ragged_batch)
    return tf_dataset


def convert_dataset(dataset, data_args, training_args, stage):
    if stage == "train":
        shuffle = True,
        batch_size = training_args.per_device_train_batch_size
        drop_remainder = True
    else:
        shuffle = False
        batch_size = training_args.per_device_eval_batch_size
        drop_remainder = False

    if isinstance(training_args.strategy, tf.distribute.TPUStrategy) or data_args.pad_to_max_length:
        logger.info(
            "Padding all batches to max length because argument was set or we're on TPU.")
        dataset_mode = "constant_batch"
    else:
        dataset_mode = "variable_batch"

    tf_data = convert_dataset_for_tensorflow(
        dataset,
        batch_size=batch_size,
        dataset_mode=dataset_mode,
        shuffle=shuffle,
        drop_remainder=drop_remainder,
    )
    return tf_data


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


def evaluate_model(model, processed_data, tf_data, data_args, output_dir=None):
    predictions = model.predict(tf_data)["logits"]
    predicted_class = np.argmax(predictions, axis=1)

    labels = processed_data["label"]
    acc = sum(predicted_class == labels) / len(labels)
    metrics = {"accuracy": acc}

    if output_dir is not None:
        id2label = {i: l for l, i in data_args.label2id.items()}
        output_file = output_dir / "test_results.txt"
        with open(output_file, "w") as writer:
            writer.write("index\tprediction\n")
            for index, item in enumerate(predicted_class):
                item = id2label[item]
                writer.write(f"{index}\t{item}\n")

    return metrics
