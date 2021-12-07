import logging
import sys
import itertools as it
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List

import numpy as np
import tensorflow as tf
from datasets import load_dataset as hf_load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TFTrainingArguments,
    set_seed,
)
from transformers.file_utils import CONFIG_NAME, TF2_WEIGHTS_NAME
from sudachitra.tokenization_bert_sudachipy import BertSudachipyTokenizer

import classification_utils
import qa_utils


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


class TaskType(Enum):
    CLASSIFICATION = "classification"
    QA = "qa"


dataset_info = {
    "amazon": {"task": TaskType.CLASSIFICATION},
    "rcqa": {"task": TaskType.QA},
}


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

    # for sudachi tokenizer
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

    dataset_dir: Optional[str] = field(
        metadata={"help": "A root directory where dataset files locate."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv file containing the training data. Overwrites dataset_dir."})
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv file containing the validation data. Overwrites dataset_dir."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv file containing the test data. Overwrites dataset_dir."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "A identifier of dataset."}
    )
    task_type: Optional[str] = field(
        default=None, metadata={"help": "Task type of dataset."}
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
            "help": "The maximum total input sequence length after tokenization. "
            "Sequences longer than this will be truncated, sequences shorter will be padded."
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

    # for qa
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

    def __post_init__(self):
        # set tasktype
        if self.dataset_name in dataset_info:
            self.task_type = dataset_info[self.dataset_name]["task"]
        else:
            if self.task_type is None:
                logger.warning(
                    f"task_type not found. Assume this is classification task.")
                self.task_type = TaskType.CLASSIFICATION
            self.task_type = TaskType(self.task_type.lower())

        # search dataset_dir for data files
        if self.dataset_dir is not None:
            data_dir = Path(self.dataset_dir)
            if self.train_file is None:
                files = list(data_dir.glob("train*"))
                self.train_file = sorted(files)[0] if len(files) > 0 else None
            if self.validation_file is None:
                files = list(data_dir.glob("dev*"))
                self.validation_file = sorted(
                    files)[0] if len(files) > 0 else None
            if self.test_file is None:
                files = list(data_dir.glob("test*"))
                self.test_file = sorted(files)[0] if len(files) > 0 else None

        # check extension of data files
        extensions = {
            self.train_file.suffix.lower() if self.train_file is not None else None,
            self.validation_file.suffix.lower() if self.validation_file is not None else None,
            self.test_file.suffix.lower() if self.test_file is not None else None,
        }
        extensions.discard(None)
        assert len(extensions) == 1, "All input files should have same extension."
        ext = extensions.pop()[1:]
        assert ext in ["csv", "json"], "data file should be csv or json."
        self.input_file_extension = ext


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


def load_dataset(data_args, training_args):
    do_step = {"train": training_args.do_train,
               "validation": training_args.do_eval,
               "test": training_args.do_predict}
    data_files = {"train": data_args.train_file,
                  "validation": data_args.validation_file,
                  "test": data_args.test_file}

    if any(do_step.values()):
        # if any of do_train/eval/predict is set, focus on them
        for k, do in do_step.items():
            if do:
                assert data_files[k] is not None, f"data file for {k} is required."
            else:
                data_files[k] = None
    else:
        # otherwise, work with provided data
        training_args.do_train = data_files["train"] is not None
        training_args.do_eval = data_files["validation"] is not None
        training_args.do_predict = data_files["test"] is not None

    data_files = {k: str(Path(v).resolve()) for k, v in data_files.items()}

    # "csv" and "json" are valid
    dataset = hf_load_dataset(
        data_args.input_file_extension, data_files=data_files)

    return dataset


def setup_tokenizer(model_args):
    if model_args.use_sudachi:
        WORD_TYPE = model_args.word_form_type
        UNIT_TYPE = model_args.split_unit_type

        word_type_token = "normalized_and_surface" if WORD_TYPE == "normalized_surface" else WORD_TYPE

        tokenizer = BertSudachipyTokenizer(
            do_lower_case=False,
            do_nfkc=True,  # default: False
            do_word_tokenize=True,
            do_subword_tokenize=True,
            vocab_file=model_args.sudachi_vocab_file,
            word_form_type=word_type_token,
            sudachipy_kwargs={
                "split_mode": UNIT_TYPE,
                "dict_type": "core",
            }
        )
    else:
        tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


def setup_config(model_args, checkpoint, data_args):
    config_path = checkpoint or model_args.config_name or model_args.model_name_or_path

    if data_args.task_type == TaskType.CLASSIFICATION:
        config = AutoConfig.from_pretrained(
            config_path,
            num_labels=len(data_args.label2id),
        )
        # add label <-> id mapping
        config.label2id = data_args.label2id
        config.id2label = {i: l for l, i in data_args.label2id.items()}
    else:
        config = AutoConfig.from_pretrained(config_path,)

    return config


def preprocess_dataset(dataset, data_args, tokenizer):
    # limit number of samples if specified
    max_samples = {
        "train": data_args.max_train_samples,
        "validation": data_args.max_val_samples,
        "test": data_args.max_test_samples,
    }
    for key in max_samples:
        if key in dataset and max_samples[key] is not None:
            dataset[key] = dataset[key].select(range(max_samples[key]))

    # tokenize sentences
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(f"max_seq_length ({data_args.max_seq_length}) is larger than the "
                       f"model maximum length ({tokenizer.model_max_length}). Use latter.")
    max_length = min(tokenizer.model_max_length, data_args.max_seq_length)

    if data_args.task_type == TaskType.CLASSIFICATION:
        dataset = classification_utils.preprocess_dataset(
            dataset, data_args, tokenizer, max_length)
    elif data_args.task_type == TaskType.QA:
        dataset = qa_utils.preprocess_dataset(
            dataset, data_args, tokenizer, max_length)
    else:
        raise ValueError(f"Unknown task type: {data_args.task_type}.")

    return dataset


def convert_to_tf_datasets(datasets, data_args, training_args):
    if data_args.task_type == TaskType.CLASSIFICATION:
        skip_keys = ()
        convert_func = classification_utils.convert_dataset_for_tensorflow
    elif data_args.task_type == TaskType.QA:
        skip_keys = ("validation", "test")
        convert_func = qa_utils.convert_dataset_for_tensorflow

    tf_datasets = {}
    for key in ("train", "validation", "test"):
        if key not in datasets or key in skip_keys:
            tf_datasets[key] = None
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

        tf_datasets[key] = convert_func(
            datasets[key],
            batch_size=batch_size,
            dataset_mode=dataset_mode,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
        )

    return tf_datasets


def setup_model(model_args, checkpoint, config, training_args, task_type):
    model_name_or_path = model_args.model_name_or_path if checkpoint is None else checkpoint
    from_pt = model_args.from_pt if checkpoint is None else False

    if task_type == TaskType.CLASSIFICATION:
        model = classification_utils.setup_model(
            model_name_or_path, config, training_args, from_pt)
    elif task_type == TaskType.QA:
        model = qa_utils.setup_model(
            model_name_or_path, config, training_args, from_pt)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

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


def finetune_model(model, tf_datasets, data_args, training_args, done_epochs):
    if data_args.task_type not in (TaskType.CLASSIFICATION, TaskType.QA):
        raise ValueError(f"Unknown task_type: {data_args.task_type}")

    rest_epochs = int(training_args.num_train_epochs) - done_epochs
    callbacks = [SaveModelCallback(training_args, done_epochs)]

    model.fit(
        tf_datasets["train"],
        validation_data=tf_datasets["validation"],
        epochs=rest_epochs,
        callbacks=callbacks,
    )
    return model


def evaluate_model(model, dataset, processed_dataset, tf_dataset, data_args, output_dir):
    if data_args.task_type == TaskType.CLASSIFICATION:
        metrics = classification_utils.evaluate_model(
            model, processed_dataset, tf_dataset, data_args, output_dir)

    elif data_args.task_type == TaskType.QA:
        metrics = qa_utils.evaluate_model(
            model, dataset, processed_dataset, data_args, output_dir)

    return metrics


def main():
    model_args, data_args, training_args = parse_args()
    logger.info(f"model args {model_args}")
    logger.info(f"data args {data_args}")
    logger.info(f"training args {training_args}")

    #
    if data_args.dataset_name == "list":
        logger.info(f"dataset_names implemented:")
        for nm in dataset_info:
            logger.info(f"{nm}")
        return
    if data_args.dataset_name not in dataset_info:
        logger.error(f"dataset_name passed ({data_args.dataset_name}) is not implemented. "
                     f"It must be one of {list(dataset_info.keys())} or \"list\".")
        return

    set_seed(training_args.seed)

    logger.info(f"load components:")
    output_root = Path(training_args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = load_dataset(data_args, training_args)
    if data_args.task_type == TaskType.CLASSIFICATION:
        # num_label is neccessary to initialize model
        dataset_key = list(datasets.keys())[0]  # at least one data file exists
        data_args.label_list = datasets[dataset_key].unique("label")
        data_args.label2id = {l: i for i, l in enumerate(data_args.label_list)}
        logger.info(
            f"work on a classification task with {len(data_args.label_list)} labels.")
    elif data_args.task_type == TaskType.QA:
        # decide columns following huggingface example
        dataset_key = list(datasets.keys())[0]  # at least one data file exists
        clm_names = datasets[dataset_key].column_names
        data_args.question_column = "question" if "question" in clm_names else clm_names[0]
        data_args.context_column = "context" if "context" in clm_names else clm_names[1]
        data_args.answer_column = "answers" if "answers" in clm_names else clm_names[2]
        logger.info(f"work on a QA task.")
    else:
        logger.error(f"Unknown task type: {data_args.task_type}.")
        return

    tokenizer = setup_tokenizer(model_args)

    logger.info(f"preprocess data:")
    processed_datasets = preprocess_dataset(datasets, data_args, tokenizer)

    with training_args.strategy.scope():
        if training_args.do_train:
            logger.info(f"finetune model:")
            for learning_rate, batch_size in it.product(data_args.learning_rate_list, data_args.batch_size_list):
                training_args.learning_rate = learning_rate
                training_args.per_device_train_batch_size = batch_size

                logger.info(
                    f"learning_rate: {learning_rate}, batch_size: {batch_size}")

                checkpoint, done_epochs = detect_checkpoint(training_args)
                rest_epochs = int(training_args.num_train_epochs) - done_epochs
                if rest_epochs <= 0:
                    logger.info(
                        f"saved model already trained {done_epochs} epochs. skip fine-tuning.")
                    continue
                if done_epochs > 0:
                    logger.info(
                        f"continue fine-tuning from epoch {done_epochs+1}")

                tf_datasets = convert_to_tf_datasets(
                    processed_datasets, data_args, training_args)

                config = setup_config(model_args, checkpoint, data_args)
                model = setup_model(model_args, checkpoint,
                                    config, training_args, data_args.task_type)
                model = finetune_model(
                    model, tf_datasets, data_args, training_args, done_epochs)

        if training_args.do_predict:
            logger.info(f"predict with test data:")
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

                tf_datasets = convert_to_tf_datasets(
                    processed_datasets, data_args, training_args)

                config = AutoConfig.from_pretrained(
                    model_path, num_labels=len(data_args.label_list))

                model = setup_model(model_args, model_path,
                                    config, training_args, data_args.task_type)
                metrics = evaluate_model(
                    model, datasets["test"], processed_datasets["test"],
                    tf_datasets["test"], data_args, output_dir=model_path)
                eval_results[dir_name] = metrics

            for hp, mts in eval_results.items():
                for key, v in mts.items():
                    logger.info(f"{hp}, {key}: {v}")

    return


if __name__ == "__main__":
    main()
