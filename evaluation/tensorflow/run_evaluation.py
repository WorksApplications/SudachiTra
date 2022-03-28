import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

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
import multiple_choice_utils
import qa_utils
import tokenizer_utils


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
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

    def __post_init__(self):
        self.use_sudachi = self.tokenizer_name.lower() == "sudachi"

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
        default=512,
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
        (ModelArguments, DataTrainingArguments, TFTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    data_files = {k: str(Path(v).resolve())
                  for k, v in data_files.items() if v is not None}

    # "csv" and "json" are valid
    dataset = hf_load_dataset(
        data_args.input_file_extension, data_files=data_files)

    return dataset


def setup_args(data_args, datasets):
    subfunc = {
        TaskType.CLASSIFICATION: classification_utils.setup_args,
        TaskType.MULTIPLE_CHOICE: multiple_choice_utils.setup_args,
        TaskType.QA: qa_utils.setup_args,
    }.get(data_args.task_type, None)

    if subfunc is None:
        raise NotImplementedError(f"task type: {data_args.task_type}.")

    return subfunc(data_args, datasets)


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


def setup_tokenizer(model_args):
    if model_args.use_sudachi:
        tokenizer = BertSudachipyTokenizer.from_pretrained(
            model_args.model_name_or_path,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name or model_args.model_name_or_path,
            use_fast=True,
        )
    return tokenizer


def setup_config(model_args, checkpoint, data_args):
    config_path = checkpoint or model_args.config_name or model_args.model_name_or_path

    if data_args.task_type == TaskType.CLASSIFICATION:
        config = AutoConfig.from_pretrained(
            config_path,
            num_labels=len(data_args.label2id),
        )
        # add label <-> id mapping
        if config.label2id is None:
            config.label2id = data_args.label2id
            config.id2label = {i: l for l, i in config.label2id.items()}
    else:
        config = AutoConfig.from_pretrained(config_path,)

    return config


def pretokenize_texts(task_type, raw_datadict, model_args, data_args):
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


def preprocess_dataset(raw_datadict, data_args, tokenizer):
    # limit number of samples if specified
    max_samples = {
        "train": data_args.max_train_samples,
        "validation": data_args.max_val_samples,
        "test": data_args.max_test_samples,
    }
    for key in max_samples:
        if key in raw_datadict and max_samples[key] is not None:
            raw_datadict[key] = raw_datadict[key].select(
                range(max_samples[key]))

    # tokenize sentences
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(f"max_seq_length ({data_args.max_seq_length}) is larger than the "
                       f"model maximum length ({tokenizer.model_max_length}). Use latter.")
    max_length = min(tokenizer.model_max_length, data_args.max_seq_length)

    subfunc = {
        TaskType.CLASSIFICATION: classification_utils.preprocess_dataset,
        TaskType.MULTIPLE_CHOICE: multiple_choice_utils.preprocess_dataset,
        TaskType.QA: qa_utils.preprocess_dataset,
    }.get(data_args.task_type, None)

    if subfunc is None:
        raise NotImplementedError(f"task type: {data_args.task_type}.")

    datadict = subfunc(raw_datadict, data_args, tokenizer, max_length)
    return datadict


def convert_to_tf_datasets(datadict, data_args, training_args):
    skip_keys = ()
    if data_args.task_type == TaskType.QA:
        # apply different conversion in evaluate step
        skip_keys = ("validation", "test")

    convert_func = {
        TaskType.CLASSIFICATION: classification_utils.convert_dataset_for_tensorflow,
        TaskType.MULTIPLE_CHOICE: multiple_choice_utils.convert_dataset_for_tensorflow,
        TaskType.QA: qa_utils.convert_dataset_for_tensorflow,
    }.get(data_args.task_type, None)

    if convert_func is None:
        raise NotImplementedError(f"task type: {data_args.task_type}.")

    tf_datasets = {}
    for key in ("train", "validation", "test"):
        if key not in datadict or key in skip_keys:
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
            datadict[key],
            batch_size=batch_size,
            dataset_mode=dataset_mode,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
        )

    return tf_datasets


def setup_model(model_args, checkpoint, config, training_args, task_type):
    model_name_or_path = model_args.model_name_or_path if checkpoint is None else checkpoint
    from_pt = model_args.from_pt if checkpoint is None else False

    setup_func = {
        TaskType.CLASSIFICATION: classification_utils.setup_model,
        TaskType.MULTIPLE_CHOICE: multiple_choice_utils.setup_model,
        TaskType.QA: qa_utils.setup_model,
    }.get(task_type, None)

    if setup_func is None:
        raise NotImplementedError(f"task type: {task_type}.")

    model = setup_func(model_name_or_path, config, training_args, from_pt)
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
    rest_epochs = int(training_args.num_train_epochs) - done_epochs
    callbacks = [SaveModelCallback(training_args, done_epochs)]

    model.fit(
        tf_datasets["train"],
        validation_data=tf_datasets["validation"],
        epochs=rest_epochs,
        callbacks=callbacks,
    )
    return model


def evaluate_model(model, raw_dataset, dataset, tf_dataset, data_args, config, output_dir, stage="test"):
    if data_args.task_type == TaskType.CLASSIFICATION:
        label2id = config.label2id or data_args.label2id
        return classification_utils.evaluate_model(
            model, dataset, tf_dataset, label2id, output_dir, stage=stage)

    elif data_args.task_type == TaskType.MULTIPLE_CHOICE:
        return multiple_choice_utils.evaluate_model(
            model, dataset, tf_dataset, output_dir, stage=stage)

    elif data_args.task_type == TaskType.QA:
        return qa_utils.evaluate_model(
            model, raw_dataset, dataset, data_args, output_dir, stage=stage)

    raise NotImplementedError(f"task type: {data_args.task_type}.")


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
    output_root = Path(training_args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_datadict = load_dataset(data_args, training_args)
    data_args = setup_args(data_args, raw_datadict)
    tokenizer = setup_tokenizer(model_args)

    logger.info(f"preprocess data:")
    raw_datadict = pretokenize_texts(
        task_type, raw_datadict, model_args, data_args)
    datadict = preprocess_dataset(raw_datadict, data_args, tokenizer)

    with training_args.strategy.scope():
        tf_datasets = convert_to_tf_datasets(
            datadict, data_args, training_args)

        if training_args.do_train:
            logger.info(f"finetune model:")
            logger.info(f"learning_rate: {training_args.learning_rate}, "
                        f"batch_size: {training_args.per_device_train_batch_size}")

            checkpoint, done_epochs = detect_checkpoint(training_args)
            rest_epochs = int(training_args.num_train_epochs) - done_epochs

            if rest_epochs <= 0:
                logger.info(
                    f"saved model already trained {done_epochs} epochs. skip fine-tuning.")
            else:
                logger.info(f"start fine-tuning from epoch {done_epochs+1}")

                config = setup_config(model_args, checkpoint, data_args)
                model = setup_model(model_args, checkpoint,
                                    config, training_args, data_args.task_type)
                model = finetune_model(
                    model, tf_datasets, data_args, training_args, done_epochs)

        if training_args.do_predict:
            logger.info(f"predict with test data:")

            step_keys = ["validation",
                         "test"] if training_args.do_eval else ["test"]
            eval_results = {k: dict() for k in step_keys}

            pref = generate_output_dir_name(training_args)
            for checkpoint in output_root.glob(f"{pref}*"):
                if not is_checkpoint(checkpoint):
                    continue

                logger.info(f"checkpoint: {checkpoint}")
                config = setup_config(None, checkpoint, data_args)
                model = setup_model(model_args, checkpoint,
                                    config, training_args, data_args.task_type)

                for step in step_keys:
                    metrics = evaluate_model(
                        model, raw_datadict[step], datadict[step], tf_datasets[step],
                        data_args, config, output_dir=checkpoint, stage=step)
                    eval_results[step][checkpoint.name] = metrics

            for step in eval_results:
                logger.info(f"evaluation result with {step} data")
                for hp, mts in eval_results[step].items():
                    for key, v in mts.items():
                        logger.info(f"{hp}, {key}: {v}")

    return


if __name__ == "__main__":
    main()
