import json
import logging
import os
import sys
from collections import defaultdict as ddict, OrderedDict as odict
from typing import Optional, Tuple

import textspan
import numpy as np
from datasets import load_metric, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerFast,
    Trainer,
    default_data_collator,
    is_torch_tpu_available,
)
from transformers.trainer_utils import PredictionOutput
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger.setLevel(logging.INFO)


def setup_args(data_args, raw_datadict):
    # decide columns following huggingface example
    dataset_key = list(raw_datadict.keys())[0]  # at least one data file exists
    clm_names = raw_datadict[dataset_key].column_names

    data_args.question_column = "question" if "question" in clm_names else clm_names[0]
    data_args.context_column = "context" if "context" in clm_names else clm_names[1]
    data_args.answer_column = "answers" if "answers" in clm_names else clm_names[2]
    data_args.column_names = clm_names
    return data_args


def pretokenize_texts(raw_datadict, pretok, data_args):
    question_column = data_args.question_column
    context_column = data_args.context_column
    answer_column = data_args.answer_column

    def subfunc(examples):
        examples[question_column] = [
            pretok(q.lstrip()) for q in examples[question_column]]
        examples[context_column] = [
            pretok(c) for c in examples[context_column]]

        if not pretok.is_identity:
            # recalculate answer_start for pretoked contexts
            answer_lists = []  # List[Dict[str, List]]
            for context, answers in zip(examples[context_column], examples[answer_column]):
                answer_lists.append({"text": [], "answer_start": []})
                context_strip, offsets = zip(
                    *[(ch, ptr) for ptr, ch in enumerate(context) if not ch.isspace()])
                context_reconcat = "".join(context_strip)
                for ans in answers["text"]:
                    ans = "".join(ch for ch in pretok(ans) if not ch.isspace())
                    idx = context_reconcat.index(ans)
                    ans_s, ans_e = offsets[idx], offsets[idx + len(ans) - 1]
                    answer_lists[-1]["text"].append(context[ans_s:ans_e+1])
                    answer_lists[-1]["answer_start"].append(ans_s)
            examples[answer_column] = answer_lists

        return examples

    raw_datadict = raw_datadict.map(subfunc, batched=True)
    return raw_datadict


def preprocess_dataset(raw_datadict, data_args, tokenizer, max_length):
    question_column = data_args.question_column
    context_column = data_args.context_column
    answer_column = data_args.answer_column

    padding = "max_length" if data_args.pad_to_max_length else False

    is_fast_tokenizer = isinstance(tokenizer, PreTrainedTokenizerFast)
    if is_fast_tokenizer:
        logger.info(f"The tokenizer is PreTrainedTokenizerFast.")
    else:
        logger.info(
            f"The tokenizer is not PreTrainedTokenizerFast. We will mimic some its features.")

    def subfunc_train(examples):
        # strip question
        examples[question_column] = [q.lstrip()
                                     for q in examples[question_column]]

        # tokenize
        # if not is_fast, result.keys = ["input_ids", "token_type_ids", "attension_mask"]
        # otherwise it also has ["offset_mapping", "overflow_sample_mapping"]
        result = tokenizer(
            examples[question_column],
            examples[context_column],
            max_length=max_length,
            truncation="only_second",
            stride=data_args.doc_stride,
            padding=padding,
            return_overflowing_tokens=is_fast_tokenizer,
            return_offsets_mapping=is_fast_tokenizer,
        )

        # when using PreTrainedTokenizerFast, one example might have multiple samples if it has a long context.
        # this is mapping from them to original example id.
        sample_mapping = result.pop(
            "overflow_to_sample_mapping", list(range(len(examples[question_column]))))

        # mapping from each tokens to their span in the original text
        offset_mapping = result.pop("offset_mapping", None)
        if offset_mapping is None:
            # construct manually, since offset_mapping is not available for PreTrainedTokenizer
            offset_mapping = _construct_offset_mapping(
                examples[question_column], examples[context_column], result, tokenizer)

        result["start_positions"] = []
        result["end_positions"] = []
        for i, offsets in enumerate(offset_mapping):
            # use position of CLS token as answer for impossible qa
            input_ids = result["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            answers = examples[answer_column][sample_mapping[i]]
            if len(answers["answer_start"]) == 0:
                # no answer, i.e. impossible to answer
                result["start_positions"].append(cls_index)
                result["end_positions"].append(cls_index)
                continue

            # only consider first answer
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # is [0, .., 0, 1, .., 1 (, 0, ..,0)], where 1 for tokens from context, 0s at last exists if padded
            token_types = result["token_type_ids"][i]
            token_start_idx = token_types.index(1)
            token_end_idx = len(token_types) - 1
            while token_types[token_end_idx] != 1:
                token_end_idx -= 1
            token_end_idx -= 1  # skip sep_token at last

            if not (offsets[token_start_idx][0] <= start_char and offsets[token_end_idx][1] >= end_char):
                result["start_positions"].append(cls_index)
                result["end_positions"].append(cls_index)
                continue

            while token_start_idx < len(offsets) and offsets[token_start_idx][0] <= start_char:
                token_start_idx += 1
            while offsets[token_end_idx][1] >= end_char:
                token_end_idx -= 1
            result["start_positions"].append(token_start_idx - 1)
            result["end_positions"].append(token_end_idx + 1)

        return result

    def subfunc_test(examples):
        # strip question
        examples[question_column] = [q.lstrip()
                                     for q in examples[question_column]]

        # tokenize
        # if not is_fast, result.keys = ["input_ids", "token_type_ids", "attension_mask"]
        # otherwise it also has ["offset_mapping", "overflow_sample_mapping"]
        result = tokenizer(
            examples[question_column],
            examples[context_column],
            max_length=max_length,
            truncation="only_second",
            stride=data_args.doc_stride,
            padding=padding,
            return_overflowing_tokens=is_fast_tokenizer,
            return_offsets_mapping=is_fast_tokenizer,
        )

        # when using PreTrainedTokenizerFast, one example might have multiple samples if it has a long context.
        # this is mapping from them to original example id.
        sample_mapping = result.pop(
            "overflow_to_sample_mapping", list(range(len(examples[question_column]))))

        # mapping from each tokens to their span in the original text
        if "offset_mapping" not in result:
            # construct manually, since offset_mapping is not available for PreTrainedTokenizer
            result["offset_mapping"] = _construct_offset_mapping(
                examples[question_column], examples[context_column], result, tokenizer)

        result["example_id"] = []
        for i, token_types in enumerate(result["token_type_ids"]):
            sample_idx = sample_mapping[i]
            result["example_id"].append(examples["id"][sample_idx])

            context_end = len(token_types) - 1
            while token_types[context_end] == 0:
                context_end -= 1
            token_types[context_end] = 0  # skip last sep_token

            # set None for spans of non-context ids
            result["offset_mapping"][i] = [
                (o if token_types[k] == 1 else None)
                for k, o in enumerate(result["offset_mapping"][i])
            ]

        return result

    processed = {}
    for key, subfunc in (("train", subfunc_train), ("validation", subfunc_test), ("test", subfunc_test)):
        if key in raw_datadict:
            processed[key] = raw_datadict[key].map(
                subfunc,
                batched=True,
                remove_columns=data_args.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
    processed = DatasetDict(processed)

    return processed


def _construct_offset_mapping(questions, contexts, tokenized, tokenizer):
    # this may fail due to unk_token or normalization process of tokenizer
    reset_token_ids = {
        tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id}
    offset_mapping = []
    for question, context, input_ids, token_types in zip(
            questions, contexts, tokenized["input_ids"], tokenized["token_type_ids"]):

        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        split_idx = token_types.index(1)
        ids_q = input_ids[:split_idx]
        ids_c = input_ids[split_idx:]
        # spans: List[List[Tuple[int, int]]]
        spans_q = textspan.get_original_spans(
            tokens[:split_idx], question)
        spans_c = textspan.get_original_spans(
            tokens[split_idx:], context)

        offsets = []  # :List[Tuple[int, int]]
        for (ids, spans) in ((ids_q, spans_q), (ids_c, spans_c)):
            assert len(ids) == len(
                spans), "the number of tokens and spans should be same"
            for i in range(len(ids)):
                if ids[i] in reset_token_ids:
                    offsets.append((0, 0))
                elif len(spans[i]) > 0:
                    offsets.append(spans[i][0])
                else:
                    # complement based on prev/next span if none found.
                    # if prev/next is null, add empty span
                    begin = 0 if i == 0 else offsets[-1][1]
                    if i+1 == len(spans) or len(spans[i+1]) == 0:
                        end = begin
                    else:
                        # begin_idx of the first span of the next spans
                        end = spans[i+1][0][0]
                    offsets.append((begin, end))
        offset_mapping.append(offsets)
    return offset_mapping


def setup_trainer(model_name_or_path, config_name, raw_datadict, datadict, data_args, training_args, tokenizer, from_tf=False):
    config = AutoConfig.from_pretrained(config_name or model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name_or_path,
        config=config,
        from_tf=from_tf,
    )

    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    metric = load_metric("squad_v2")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = _postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=True,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]

        references = [{"id": ex["id"], "answers": ex[data_args.answer_column]}
                      for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=datadict["train"] if training_args.do_train else None,
        eval_dataset=datadict["validation"] if training_args.do_eval else None,
        eval_examples=raw_datadict["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
        post_process_function=post_processing_function,
    )
    return trainer


def evaluate_model(trainer, raw_dataset, dataset, output_dir=None, stage="eval"):
    results = trainer.predict(dataset, raw_dataset)

    id2answers = {d["id"]: d["answers"]["text"] for d in results.label_ids}
    if output_dir is not None:
        output_file = output_dir / f"{stage}_predictions.tsv"
        if trainer.is_world_process_zero():
            with open(output_file, "w") as w:
                w.write("id\tanswer\tprediction\n")
                for p in results.predictions:
                    did = p["id"]
                    ans = id2answers[did]
                    ans = "" if len(ans) == 0 else ans[0]
                    pred = p["prediction_text"]
                    w.write(f"{did}\t{ans}\t{pred}\n")

    metrics = results.metrics
    trainer.log_metrics(stage, metrics)
    trainer.save_metrics(stage, metrics)
    return metrics


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(
            predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)


def _postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
):
    """taken from https://github.com/huggingface/transformers/blob/master/examples/tensorflow/question-answering/utils_qa.py"""

    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = ddict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(
            i)

    # The dictionaries we have to fill.
    all_predictions = odict()
    all_nbest_json = odict()
    if version_2_with_negative:
        scores_diff_json = odict()

    # Logging.
    logger.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(
                start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(
                end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[
            :n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            for i in range(len(predictions)):
                if predictions[i]["text"] != "":
                    break
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - \
                best_non_null_pred["start_logit"] - \
                best_non_null_pred["end_logit"]
            # To be JSON-serializable.
            scores_diff_json[example["id"]] = float(score_diff)
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v)
             for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions
