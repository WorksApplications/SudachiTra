import logging
import sys
import itertools as it
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List

import numpy as np
from transformers.trainer_utils import get_last_checkpoint
import tensorflow as tf
from datasets import load_dataset as hf_load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.file_utils import CONFIG_NAME, MULTIPLE_CHOICE_DUMMY_INPUTS, TF2_WEIGHTS_NAME
from sudachitra.tokenization_bert_sudachipy import BertSudachipyTokenizer

import classification_utils
import multiple_choice_utils
import qa_utils
import tokenizer_utils


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


class TaskType(Enum):
    CLASSIFICATION = "classification"
    MULTIPLE_CHOICE = "multiple-choice"
    QA = "qa"


dataset_info = {
    "amazon": {"task": TaskType.CLASSIFICATION},
    "kuci": {"task": TaskType.MULTIPLE_CHOICE},
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
    pretokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Tokenizer to convert text space-separated before preprocess. "
                                "\"juman\" (for Kyoto-U BERT) or \"mecab-juman\" (for NICT-BERT)."}
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
        self.use_sudachi = self.tokenizer_name.lower() == "sudachi"
        if self.use_sudachi:
            assert self.sudachi_vocab_file is not None, "sudachi_vocab_file is required to use sudachi tokenizer"
            assert self.word_form_type is not None, "word_form_type is required to use sudachi tokenizer"
            assert self.split_unit_type is not None, "split_unit_type is required to use sudachi tokenizer"

        if self.pretokenizer_name is not None:
            pretok_list = ["identity", "juman", "mecab-juman"]
            assert self.pretokenizer_name.lower() in pretok_list, \
                f"pretokenizer_name should be one of {pretok_list}"


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
        default=None, metadata={"help": f"Task type of dataset. One of {[t.value for t in TaskType]}. "
                                f"Inferred from dataset_name if not set."}
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )

    max_seq_length: int = field(
        default=384,
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
        dataset_name = self.dataset_name.lower()
        if dataset_name in dataset_info:
            self.task_type = dataset_info[dataset_name]["task"]
        else:
            if self.task_type is None:
                raise ValueError(
                    f"task_type not set and cannot infer from dataset_name.")
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
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args


def detect_checkpoint(training_args):
    checkpoint = None
    output_dir = Path(training_args.output_dir)
    if output_dir.is_dir() and training_args.do_train and not training_args.overwrite_output_dir:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and len(list(output_dir.glob("*"))) > 0:
            raise ValueError(f"output_dir ({output_dir}) exists and not empty. "
                             f"Set --overwrite_output_dir to continue anyway.")
        if checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"resume training from detected checkpoint ({checkpoint}). "
                        f"Set --overwrite_output_dir to train from scratch.")
    return checkpoint


def load_dataset(data_args, training_args):
    # load neccessary data from
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
        training_args.do_train = bool(data_files["train"] is not None)
        training_args.do_eval = bool(data_files["validation"] is not None)
        training_args.do_predict = bool(data_files["test"] is not None)

    data_files = {k: str(Path(v).resolve())
                  for k, v in data_files.items() if v is not None}

    dataset = hf_load_dataset(
        data_args.input_file_extension, data_files=data_files)

    return dataset


def setup_args(task_type, data_args, raw_datadict):
    # set task specific args
    subfunc = {
        TaskType.CLASSIFICATION: classification_utils.setup_args,
        TaskType.MULTIPLE_CHOICE: multiple_choice_utils.setup_args,
        TaskType.QA: qa_utils.setup_args,
    }.get(task_type, None)

    if subfunc is None:
        raise NotImplementedError(f"task type: {task_type}.")

    return subfunc(data_args, raw_datadict)


def setup_pretokenizer(model_args):
    # tokenizer for some models requires texts to be space-separated.
    # pretokenizer works for that.
    if model_args.pretokenizer_name == "juman":
        logger.info("Use juman for pretokenize")
        return tokenizer_utils.Juman()
    if model_args.pretokenizer_name == "mecab-juman":
        logger.info("Use mecab-juman for pretokenize")
        return tokenizer_utils.MecabJuman()

    logger.info("Skip pretokenize")
    return tokenizer_utils.Identity()


def pretokenize_texts(task_type, raw_datadict, model_args, data_args):
    # tokenize input text for some models
    pretok = setup_pretokenizer(model_args)
    if pretok.is_identity:
        return raw_datadict

    subfunc = {
        TaskType.CLASSIFICATION: classification_utils.pretokenize_texts,
        TaskType.MULTIPLE_CHOICE: multiple_choice_utils.pretokenize_texts,
        TaskType.QA: qa_utils.pretokenize_texts,
    }.get(task_type, None)

    if subfunc is None:
        raise NotImplementedError(f"task type: {task_type}")

    raw_datadict = subfunc(raw_datadict, pretok, data_args)
    return raw_datadict


def setup_tokenizer(model_args):
    if model_args.use_sudachi:
        WORD_TYPE = model_args.word_form_type
        UNIT_TYPE = model_args.split_unit_type
        word_type_token = "normalized_and_surface" if WORD_TYPE == "normalized_surface" else WORD_TYPE
        logger.info(
            f"Use sudachi tokenizer ({WORD_TYPE}, {UNIT_TYPE}, {model_args.sudachi_vocab_file}).")

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
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name or model_args.model_name_or_path,
            use_fast=True,
        )
    return tokenizer


def preprocess_dataset(task_type, raw_datadict, data_args, tokenizer):
    # limit the number of samples if specified
    max_samples = {
        "train": data_args.max_train_samples,
        "validation": data_args.max_val_samples,
        "test": data_args.max_test_samples,
    }
    for key, v in max_samples.items():
        if key in raw_datadict and v is not None:
            raw_datadict[key] = raw_datadict[key].select(range(v))

    # preprocess (apply tokenizer, etc.)
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(f"max_seq_length ({data_args.max_seq_length}) is larger than the "
                       f"model maximum length ({tokenizer.model_max_length}). Use latter.")
    max_length = min(tokenizer.model_max_length, data_args.max_seq_length)

    subfunc = {
        TaskType.CLASSIFICATION: classification_utils.preprocess_dataset,
        TaskType.MULTIPLE_CHOICE: multiple_choice_utils.preprocess_dataset,
        TaskType.QA: qa_utils.preprocess_dataset,
    }.get(task_type, None)

    if subfunc is None:
        raise NotImplementedError(f"task type: {task_type}.")

    datadict = subfunc(raw_datadict, data_args, tokenizer, max_length)
    return datadict


def setup_trainer(task_type, model_args, data_args, training_args, raw_datadict, datadict, tokenizer):
    model_name = model_args.model_name_or_path
    config_name = model_args.config_name

    if task_type == TaskType.CLASSIFICATION:
        return classification_utils.setup_trainer(
            model_name, config_name, datadict, data_args, training_args, tokenizer)
    elif task_type == TaskType.MULTIPLE_CHOICE:
        return multiple_choice_utils.setup_trainer(
            model_name, config_name, datadict, data_args, training_args, tokenizer)
    elif task_type == TaskType.QA:
        return qa_utils.setup_trainer(
            model_name, config_name, raw_datadict, datadict, data_args, training_args, tokenizer)

    raise NotImplementedError(f"task type: {task_type}.")


def train_model(trainer, checkpoint):
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics

    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    return


def evaluate_model(task_type, trainer, raw_dataset, dataset, data_args, output_dir, stage="test"):
    if task_type == TaskType.CLASSIFICATION:
        label2id = data_args.label2id
        return classification_utils.evaluate_model(
            trainer, dataset, label2id, output_dir, stage=stage)

    elif task_type == TaskType.MULTIPLE_CHOICE:
        return multiple_choice_utils.evaluate_model(
            trainer, dataset, output_dir, stage=stage)

    elif task_type == TaskType.QA:
        return qa_utils.evaluate_model(
            trainer, raw_dataset, dataset, output_dir, stage=stage)

    raise NotImplementedError(f"task type: {task_type}.")


def main():
    model_args, data_args, training_args = parse_args()
    logger.info(f"model args {model_args}")
    logger.info(f"data args {data_args}")
    logger.info(f"training args {training_args}")

    if data_args.dataset_name.lower() == "list":
        logger.info(f"dataset_names implemented:")
        for nm in dataset_info:
            logger.info(f"{nm}")
        return
    if data_args.dataset_name.lower() not in dataset_info:
        logger.error(f"dataset_name passed ({data_args.dataset_name}) is not implemented. "
                     f"It must be one of {list(dataset_info.keys())} or \"list\".")
        return

    set_seed(training_args.seed)
    task_type = data_args.task_type

    logger.info(f"load components:")
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # run first to raise error early
    checkpoint = detect_checkpoint(training_args)

    raw_datadict = load_dataset(data_args, training_args)
    data_args = setup_args(task_type, data_args, raw_datadict)
    tokenizer = setup_tokenizer(model_args)

    logger.info(f"preprocess data:")
    raw_datadict = pretokenize_texts(
        task_type, raw_datadict, model_args, data_args)
    datadict = preprocess_dataset(
        task_type, raw_datadict, data_args, tokenizer)

    trainer = setup_trainer(task_type, model_args, data_args,
                            training_args, raw_datadict, datadict, tokenizer)

    if training_args.do_train:
        logger.info("step: train")
        checkpoint = training_args.resume_from_checkpoint or checkpoint
        train_model(trainer, checkpoint)

    step_keys = []
    if training_args.do_eval:
        step_keys.append("validation")
    if training_args.do_predict:
        step_keys.append("test")

    eval_results = {k: dict() for k in step_keys}
    for step in step_keys:
        logger.info(f"step: {step}")
        metrics = evaluate_model(
            task_type, trainer, raw_datadict[step], datadict[step], data_args, output_dir, stage=step)
        eval_results[step] = metrics

    logger.info(f"result map: {eval_results}")
    return


if __name__ == "__main__":
    main()
