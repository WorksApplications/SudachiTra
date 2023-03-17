import itertools as it
import logging
import sys
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    EvalPrediction,
    Trainer,
    default_data_collator,
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger.setLevel(logging.INFO)


def setup_args(data_args, raw_datadict):
    # set dataset column name, assuming to use convert_dataset.py
    dataset_key = list(raw_datadict.keys())[0]  # at least one data file exists
    column_names = raw_datadict[dataset_key].column_names

    data_args.context_column = "context"
    data_args.choice_columns = [
        c for c in column_names if c.startswith("choice")]
    data_args.label_column = "label"
    return data_args


def pretokenize_texts(raw_datadict, pretok, data_args):
    text_columns = data_args.choice_columns + [data_args.context_column]

    def subfunc(examples):
        for c in text_columns:
            examples[c] = [pretok(s) for s in examples[c]]
        return examples

    raw_datadict = raw_datadict.map(subfunc, batched=True)
    return raw_datadict


def preprocess_dataset(raw_datadict, data_args, tokenizer, max_length):
    context_column = data_args.context_column
    choice_columns = data_args.choice_columns
    n_choices = len(choice_columns)

    padding = "max_length" if data_args.pad_to_max_length else False

    def subfunc(examples):
        first_sentences = ([c] * n_choices for c in examples[context_column])
        second_sentences = (examples[clm] for clm in choice_columns)

        # flatten
        first_sentences = list(it.chain(*first_sentences))
        second_sentences = list(it.chain(*zip(*second_sentences)))

        # tokenize
        tokenized = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=max_length,
            padding=padding,
        )

        # un-flatten
        result = {k: [v[i:i+n_choices] for i in range(0, len(v), n_choices)]
                  for k, v in tokenized.items()}

        # keep label column as it is, assuming it contains 0-indexed integer
        return result

    datadict = raw_datadict.map(
        subfunc,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    return datadict


def setup_trainer(model_name_or_path, config_name, datadict, data_args, training_args, tokenizer, from_tf=False):
    config = AutoConfig.from_pretrained(config_name or model_name_or_path)
    model = AutoModelForMultipleChoice.from_pretrained(
        model_name_or_path,
        config=config,
        from_tf=from_tf,
    )

    def compute_metrics(p: EvalPrediction):
        predictions, label_ids = p
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datadict["train"] if training_args.do_train else None,
        eval_dataset=datadict["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    return trainer


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(it.chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1)
                 for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def evaluate_model(trainer, dataset, output_dir=None, stage="eval"):
    p = trainer.predict(dataset, metric_key_prefix=stage)
    predictions = np.argmax(p.predictions, axis=1)
    labels = p.label_ids if p.label_ids is not None else dataset["label"]
    metrics = p.metrics if p.metrics is not None else {}

    if output_dir is not None:
        output_file = output_dir / f"{stage}_predictions.tsv"
        if trainer.is_world_process_zero():
            with open(output_file, "w") as w:
                w.write("index\tlabel\tprediction\n")
                for i, (l, p) in enumerate(zip(labels, predictions)):
                    w.write(f"{i}\t{l}\t{p}\n")

    trainer.log_metrics(stage, metrics)
    trainer.save_metrics(stage, metrics)
    return metrics
