import logging
import sys

import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    EvalPrediction,
    Trainer,
    default_data_collator,
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

    data_args.label_list = sorted(label_list)  # sort for determinism
    data_args.label2id = {l: i for i, l in enumerate(label_list)}

    # columns of input text (2 columns maximum)
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
    padding = "max_length" if data_args.pad_to_max_length else False

    # Truncate text before tokenization for sudachi, which has a input bytes limit.
    # This may affect the result with a large max_length (tokens).
    MAX_CHAR_LENGTH = 2**14

    def subfunc(examples):
        # Tokenize texts
        texts = ([s[:MAX_CHAR_LENGTH] for s in examples[c]]
                 for c in data_args.text_columns)
        result = tokenizer(*texts, padding=padding,
                           max_length=max_length, truncation=True)

        # Map labels to ids
        if "label" in examples:
            result["label"] = [
                (data_args.label2id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datadict = raw_datadict.map(
        subfunc,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        remove_columns=data_args.data_columns,
    )
    return datadict


def setup_config(config_name_or_path, data_args):
    config = AutoConfig.from_pretrained(
        config_name_or_path,
        finetuning_task=data_args.dataset_name,
        num_labels=len(data_args.label2id),
    )
    # add label <-> id mapping
    config.label2id = data_args.label2id
    config.id2label = {i: l for l, i in config.label2id.items()}
    return config


def setup_trainer(model_name_or_path, config_name, datadict, data_args, training_args, tokenizer, from_tf=False):
    config = setup_config(config_name or model_name_or_path, data_args)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        from_tf=from_tf,
    )

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(
            p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datadict["train"] if training_args.do_train else None,
        eval_dataset=datadict["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer


def evaluate_model(trainer, dataset, label2id=lambda x: x, output_dir=None, stage="eval"):
    p = trainer.predict(dataset, metric_key_prefix=stage)
    predictions = np.argmax(p.predictions, axis=1)
    labels = p.label_ids if p.label_ids is not None else dataset["label"]
    metrics = p.metrics if p.metrics is not None else {}

    if output_dir is not None:
        i2l = {i: l for l, i in label2id.items()}
        output_file = output_dir / f"{stage}_predictions.tsv"
        if trainer.is_world_process_zero():
            with open(output_file, "w") as w:
                w.write("index\tlabel\tprediction\n")
                for i, (l, p) in enumerate(zip(labels, predictions)):
                    w.write(f"{i}\t{i2l[l]}\t{i2l[p]}\n")

    trainer.log_metrics(stage, metrics)
    trainer.save_metrics(stage, metrics)
    return metrics
