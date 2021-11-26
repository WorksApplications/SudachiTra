import logging
import sys
import itertools as it
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np
import tensorflow as tf
from datasets import load_dataset as load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TFTrainingArguments,
    TFAutoModelForSequenceClassification,
    set_seed,
)
from transformers.file_utils import CONFIG_NAME, TF2_WEIGHTS_NAME
from sudachitra.tokenization_bert_sudachipy import BertSudachipyTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


@dataclass
class ModelArguments:
    """
    Arguments for model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "\"sudachi\" or pretrained tokenizer name or path if not the same as model_name"}
    )
    from_pt: bool = field(
        default=False, metadata={"help": "Set True when load PyTorch save file"}
    )

    sudachi_vocab_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the sudachi tokenizer vocab file. Required if use sudachi"}
    )
    word_form_type: Optional[str] = field(
        default=None, metadata={"help": "Word form type for sudachi tokenizer: surface/normalized/normalized_surface. Required if use sudachi"}
    )
    split_unit_type: Optional[str] = field(
        default=None, metadata={"help": "Split unit type for sudachi tokenizer: A/B/C. Required if use sudachi"}
    )

    def __post_init__(self):
        self.use_sudachi = self.tokenizer_name in ["sudachi", "Sudachi"]
        if self.use_sudachi:
            assert self.sudachi_vocab_file is not None, "sudachi_vocab_file is required to use sudachi tokenizer"
            assert self.word_form_type is not None, "word_form_type is required to use sudachi tokenizer"
            assert self.split_unit_type is not None, "split_unit_type is required to use sudachi tokenizer"


@dataclass
class DataTrainingArguments:
    """
    Arguments for data to use fine-tune and eval.
    """

    train_file: str = field(
        metadata={"help": "A csv file containing the training data."})
    test_file: str = field(
        metadata={"help": "A csv file containing the test data."})
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv file containing the validation data."}
    )

    learning_rate_list: Optional[List[float]] = field(
        default=None, metadata={"help": "The list of learning rate for hyper-parameter search. "
                                "Overrides learning_rate."}
    )
    batch_size_list: Optional[List[int]] = field(
        default=None, metadata={"help": "The list of training batch size for hyper-parameter search. "
                                "Overrides per_device_train_batch_size."}
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            "Data will always be padded when using TPUs."
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this value if set."
        },
    )

    def __post_init__(self):
        train_extension = self.train_file.split(".")[-1].lower()
        validation_extension = (
            self.validation_file.split(
                ".")[-1].lower() if self.validation_file is not None else None
        )
        test_extension = self.test_file.split(".")[-1].lower()
        extensions = {train_extension, validation_extension, test_extension}
        extensions.discard(None)
        assert {"csv"} == extensions, "All input files should be csv"
        self.input_file_extension = extensions.pop()


def parse_args():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TFTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # hyper parameter to search
    if data_args.learning_rate_list is None:
        data_args.learning_rate_list = [training_args.learning_rate]
    if data_args.batch_size_list is None:
        data_args.batch_size_list = [training_args.per_device_train_batch_size]

    return model_args, data_args, training_args


def generate_output_dir_name(training_args, n_epoch=None):
    lr = training_args.learning_rate
    bs = training_args.per_device_train_batch_size

    if n_epoch is None:
        # should be used as prefix
        return f"{lr}_{bs}"
    else:
        return f"{lr}_{bs}_{n_epoch}"


def extruct_num_epoch(dir_name_or_path):
    if dir_name_or_path is None:
        return 0

    dir_name = str(dir_name_or_path).split("/")[-1]
    n_epoch = int(dir_name.split("_")[-1])
    return n_epoch


def is_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / CONFIG_NAME).is_file() and (path / TF2_WEIGHTS_NAME).is_file():
        return True


def detect_checkpoint(training_args):  # -> Optional[Path], int
    output_root = Path(training_args.output_dir)

    pref = generate_output_dir_name(training_args)
    dirs = [p for p in output_root.glob(f"{pref}*") if p.is_dir()]

    if len(dirs) == 0 or training_args.overwrite_output_dir:
        return None, 0

    ckpt_dir = sorted(dirs)[-1]
    if is_checkpoint(ckpt_dir):
        done_epochs = extruct_num_epoch(ckpt_dir)
        logger.info(f"Checkpoint detected in {ckpt_dir}")
        return ckpt_dir, done_epochs

    raise ValueError(
        f"Directory ({ckpt_dir}) seems not checkpoint. Set overwrite_output_dir True to continue regardless")


def load_csv_dataset(data_args, training_args):
    data_files = dict()
    if training_args.do_train:
        if data_args.train_file is None:
            raise ValueError(f"train_file is neccessary to train")
        data_files["train"] = data_args.train_file
    if training_args.do_eval:
        if data_args.validation_file is None:
            raise ValueError(f"validation_file is neccessary to eval")
        data_files["validation"] = data_args.validation_file
    if training_args.do_predict:
        if data_args.test_file is None:
            raise ValueError(f"test_file is neccessary to predict")
        data_files["test"] = data_args.test_file

    data_files = {k: str(Path(v).resolve())
                  for k, v in data_files.items() if v is not None}
    dataset = load_dataset("csv", data_files=data_files)

    # limit num_sample if specified
    max_samples = {
        "train": data_args.max_train_samples,
        "validation": data_args.max_val_samples,
        "test": data_args.max_test_samples,
    }
    for key in max_samples:
        if key in dataset and max_samples[key] is not None:
            dataset[key] = dataset[key].select(range(max_samples[key]))

    return dataset


def setup_tokenizer(model_args):
    if model_args.use_sudachi:
        WORD_TYPE = model_args.word_form_type
        UNIT_TYPE = model_args.split_unit_type

        word_type_token = "normalized_and_surface" if WORD_TYPE == "normalized_surface" else WORD_TYPE

        tokenizer = BertSudachipyTokenizer(
            vocab_file=model_args.sudachi_vocab_file,
            word_form_type=word_type_token,
            sudachipy_kwargs={"split_mode": UNIT_TYPE}
        )
    else:
        tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


def setup_config(model_args, checkpoint, label2id):
    if checkpoint is not None:
        config_path = checkpoint
    elif model_args.config_name is not None:
        config_path = model_args.config_name
    else:
        config_path = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(
        config_path,
        num_labels=len(label2id),
    )

    # setup label <-> id mapping
    config.label2id = label2id
    config.id2label = {i: l for l, i in label2id.items()}

    return config


def preprocess_dataset(dataset, column_names, tokenizer, max_seq_length, label2id=lambda x: x):
    def subfunc(examples):
        # Tokenize texts (2 columns maximum)
        texts = (examples[c]
                 for c in [l for l in column_names if l != "label"][:2])
        max_length = min(tokenizer.model_max_length, max_seq_length)
        result = tokenizer(
            *texts, max_length=max_length, truncation=True)

        # Map labels to ids
        if "label" in examples:
            result["label"] = [
                (label2id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    dataset = dataset.map(subfunc, batched=True)
    return dataset


def convert_dataset_for_tensorflow(
    dataset, ignore_column_names, batch_size, dataset_mode="variable_batch", shuffle=True, drop_remainder=True
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
    feature_keys = list(set(dataset.features.keys()) -
                        set(ignore_column_names + ["label"]))
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


def convert_datasets(dataset, ignore_column_names, data_args, training_args):
    tf_data = dict()
    for key in ("train", "validation", "test"):
        if key not in dataset:
            tf_data[key] = None
            continue
        if key == "train":
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

        tf_data[key] = convert_dataset_for_tensorflow(
            dataset[key],
            ignore_column_names=ignore_column_names,
            shuffle=shuffle,
            dataset_mode=dataset_mode,
            batch_size=batch_size,
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


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, training_args, done_epochs, **kwargs):
        super().__init__()
        self.output_root = Path(training_args.output_dir)
        self.dirname_prefix = generate_output_dir_name(training_args)
        self.done_epochs = done_epochs

    def on_epoch_end(self, epoch, logs=None):
        dirname = f"{self.dirname_prefix}_{self.done_epochs+epoch+1}"
        self.model.save_pretrained(self.output_root / dirname)


def finetune_model(model, tf_data, training_args, done_epochs):
    rest_epochs = int(training_args.num_train_epochs) - done_epochs
    if rest_epochs <= 0:
        logger.info(
            f"saved model already trained {done_epochs} epochs. skip fine-tuning.")
        return model
    if done_epochs > 0:
        logger.info(f"continue fine-tuning from epoch {done_epochs+1}")

    callbacks = [SaveModelCallback(training_args, done_epochs)]
    model.fit(
        tf_data["train"],
        validation_data=tf_data["validation"],
        epochs=rest_epochs,
        callbacks=callbacks,
    )
    return model


def predict_testdata(model, tf_data):
    predictions = model.predict(tf_data)["logits"]
    predicted_class = np.argmax(predictions, axis=1)
    return predictions, predicted_class


def save_prediction(predicted_class, output_file, id2label=lambda x: x):
    with open(output_file, "w") as writer:
        writer.write("index\tprediction\n")
        for index, item in enumerate(predicted_class):
            item = id2label[item]
            writer.write(f"{index}\t{item}\n")
    return


def main():
    model_args, data_args, training_args = parse_args()
    logger.info(f"model args {model_args}")
    logger.info(f"data args {data_args}")
    logger.info(f"training args {training_args}")

    logger.info(f"load components:")
    output_root = Path(training_args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = setup_tokenizer(model_args)
    dataset = load_csv_dataset(data_args, training_args)

    logger.info(f"preprocess data:")
    # create labelset. assume that data files has same column names
    column_names = dataset.column_names["train"]
    label_list = dataset["test"].unique("label")
    label2id = {l: i for i, l in enumerate(label_list)}

    dataset = preprocess_dataset(
        dataset, column_names, tokenizer, data_args.max_seq_length, label2id)

    if training_args.do_train:
        for learning_rate, batch_size in it.product(data_args.learning_rate_list, data_args.batch_size_list):
            training_args.learning_rate = learning_rate
            training_args.per_device_train_batch_size = batch_size

            logger.info(
                f"finetune model: learning_rate: {learning_rate}, batch_size: {batch_size}")

            checkpoint, done_epochs = detect_checkpoint(training_args)
            config = setup_config(model_args, checkpoint, label2id)
            model_path = model_args.model_name_or_path if checkpoint is None else checkpoint

            with training_args.strategy.scope():
                set_seed(training_args.seed)
                tf_data = convert_datasets(
                    dataset, column_names, data_args, training_args)
                model = setup_model(model_path, config,
                                    training_args, model_args.from_pt)
                model = finetune_model(
                    model, tf_data, training_args, done_epochs)

    if training_args.do_predict:
        logger.info(f"predict with each models")
        tf_data = convert_datasets(
            dataset, column_names, data_args, training_args)
        eval_results = dict()
        for learning_rate, batch_size, n_epoch in it.product(
                data_args.learning_rate_list, data_args.batch_size_list, range(int(training_args.num_train_epochs))):
            training_args.learning_rate = learning_rate
            training_args.per_device_train_batch_size = batch_size

            dir_name = generate_output_dir_name(training_args, n_epoch+1)
            model_path = output_root / dir_name
            if not model_path.exists():
                logger.info(f"model {dir_name} does not found. "
                            f"run with --do_train option to train model")
                continue

            config = AutoConfig.from_pretrained(
                model_path, num_labels=len(label_list))

            with training_args.strategy.scope():
                model = setup_model(model_path, config, training_args)
                _, predicted_class = predict_testdata(model, tf_data["test"])
                eval_results[dir_name] = predicted_class
                save_prediction(predicted_class, model_path /
                                "test_results.txt", config.id2label)

        labels = dataset["test"]["label"]
        for key, predicted_class in eval_results.items():
            acc = sum(predicted_class == labels) / len(labels)
            print(f"{key}, acc: {acc:.4f}")

    return


if __name__ == "__main__":
    main()
