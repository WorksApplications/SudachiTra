import numpy as np
import tensorflow as tf
from transformers import (
    TFAutoModelForSequenceClassification,
)


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
