import argparse as ap
import logging
import sys
from collections import defaultdict as ddict
from pathlib import Path
from typing import List, Dict

from sudachitra.input_string_normalizer import InputStringNormalizer
from sudachitra.word_formatter import word_formatter, WordFormTypes
from sudachipy import Dictionary, SplitMode
import tokenizer_utils
from datasets import load_dataset, Dataset, DatasetDict


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger.setLevel(logging.INFO)


def convert_amazon(args):
    # convert Amazon Review Corpus (https://registry.opendata.aws/amazon-reviews-ml/)

    if args.dataset_dir_or_file is not None:
        logger.warning(
            f"Load data from hugging face hub and ignore --input arg.")
    else:
        logger.info("Load data from hugging face hub.")

    datadict = load_dataset("amazon_reviews_multi", "ja")

    if args.seed is not None or args.split_rate is not None:
        logger.warning("amazon_reviews_multi dataset is already splitted. "
                       "skip shuffle and split")

    datadict = select_column(datadict, {
        "review_body": "sentence1",
        "stars": "label",
    })
    datadict = DatasetDict({
        "train": datadict["train"],
        "dev": datadict["validation"],
        "test": datadict["test"]}
    )

    text_columns = ["sentence1"]
    datadict = normalize_texts(datadict, text_columns, args.word_form)
    datadict = tokenize_texts(datadict, text_columns,
                              args.tokenizer, args.dicdir, args.mecabrc)
    return datadict


def convert_kuci(args):
    # convert 京都大学常識推論データセット(KUCI) (https://nlp.ist.i.kyoto-u.ac.jp/?KUCI)

    if args.dataset_dir_or_file is None:
        raise ValueError(f"Provide the raw data directory with --input option. "
                         f"Download and untar it first (https://nlp.ist.i.kyoto-u.ac.jp/?KUCI).")
    dataset_dir = Path(args.dataset_dir_or_file)
    if not dataset_dir.exists():
        raise ValueError(f"{dataset_dir} does not exists.")
    if dataset_dir.is_file():
        raise ValueError(
            f"Provide untared directory instead of tar file: {dataset_dir}.")

    datafiles = {
        "train": str(dataset_dir / "train.jsonl"),
        "dev": str(dataset_dir / "development.jsonl"),
        "test": str(dataset_dir / "test.jsonl"),
    }
    for p in datafiles.values():
        if not Path(p).exists():
            raise ValueError(f"File {p} doen not exists.")

    datadict = load_dataset("json", data_files=datafiles)

    if args.seed is not None or args.split_rate is not None:
        logger.warning("KUCI dataset is already splitted. "
                       "skip shuffle and split")

    a2i = {c: i for i, c in enumerate("abcd")}

    def convert(example):
        # change choice index from alphabet to integer
        example["label"] = [a2i[a] for a in example["label"]]

        # concatenate tokens (raw texts are tokenized using Juman)
        example["context"] = ["".join(t.split()) for t in example["context"]]
        for a, i in a2i.items():
            example[f"choice_{i}"] = ["".join(t.split())
                                      for t in example[f"choice_{a}"]]
        return example

    datadict = datadict.map(convert, batched=True,
                            remove_columns=[f"choice_{a}" for a in a2i.keys()])

    text_columns = ["context"] + [f"choice_{i}" for i in a2i.values()]
    datadict = normalize_texts(datadict, text_columns, args.word_form)
    datadict = tokenize_texts(datadict, text_columns,
                              args.tokenizer, args.dicdir, args.mecabrc)
    return datadict


def convert_livedoor(args):
    raise NotImplementedError()


def convert_rcqa(args):
    # convert 解答可能性付き読解データセット (http://www.cl.ecei.tohoku.ac.jp/rcqa/),
    # following NICT's experiment (https://alaginrc.nict.go.jp/nict-bert/Experiments_on_RCQA.html).
    # also refer: https://github.com/tsuchm/nict-bert-rcqa-test

    if args.dataset_dir_or_file is None:
        raise ValueError(f"Provide the raw data file with --input option. "
                         f"Download it first (http://www.cl.ecei.tohoku.ac.jp/rcqa/all-v1.0.json.gz).")

    dataset_file = Path(args.dataset_dir_or_file)
    FILE_NAME = "all-v1.0.json.gz"
    if dataset_file.is_dir():
        dataset_file = dataset_file / FILE_NAME
    if not dataset_file.exists():
        raise ValueError(f"File {dataset_file} does not exists.")

    dataset = load_dataset("json", data_files=str(dataset_file))["train"]

    if args.split_rate is None:
        logger.info("split data by timestamp, following NICT's experiment")
        datadict = DatasetDict({
            "train": dataset.filter(lambda row: row["timestamp"] < "2009"),
            "dev": dataset.filter(lambda row: "2009" <= row["timestamp"] < "2010"),
            "test": dataset.filter(lambda row: "2010" <= row["timestamp"]),
        })
    else:
        dataset = shuffle_dataset(dataset, args.seed)
        datadict = split_dataset(dataset, args.split_rate)

    # flatten documents column
    def flatten_doc(examples):
        outputs = ddict(list)
        clms = [c for c in examples if c != "documents"]
        for i, docs in enumerate(examples["documents"]):
            for c in clms:
                outputs[c].extend((examples[c][i] for _ in range(len(docs))))
            outputs["documents"].extend(docs)
            outputs["doc_id"].extend(range(1, len(docs)+1))
        return outputs
    datadict = datadict.map(flatten_doc, batched=True).flatten()

    if args.word_form != WordFormTypes.SURFACE:
        # normalize texts manually, to handle answer properly
        sudachi_dict = Dictionary(dict="core")
        sudachi = sudachi_dict.create(SplitMode.C)
        normalizer = InputStringNormalizer(do_lower_case=False, do_nfkc=True)
        formatter = word_formatter(args.word_form, sudachi_dict)

        def normalize(t): return "".join(
            formatter(m) for m in sudachi.tokenize(normalizer.normalize_str(t)))

        def convert(ex):
            ex["question"] = normalize(ex["question"])

            morphs = [(m.surface(), formatter(m)) for m in sudachi.tokenize(
                normalizer.normalize_str(ex["documents.text"]))]
            ex["documents.text"] = "".join(m[1] for m in morphs)

            # search answer
            answer = ex["answer"]
            answer_nrm = normalize(answer)
            if answer in ex["documents.text"]:
                ex["answer"] = answer
                return ex
            if answer_nrm in ex["documents.text"]:
                ex["answer"] = answer_nrm
                return ex

            # answer and corresponding context substring normalized differently
            # this might output wrong str depending on tokenization
            for ib in range(len(morphs)):
                if answer not in "".join(m[0] for m in morphs[ib:]):
                    ib -= 1
                    break
            for ie in range(ib+1, len(morphs)+1):
                if answer in "".join(m[0] for m in morphs[ib:ie]):
                    break
            ex["answer"] = "".join(m[1] for m in morphs[ib:ie])
            return ex

        datadict = datadict.map(convert)

    text_columns = ["question", "documents.text", "answer"]
    datadict = tokenize_texts(datadict, text_columns,
                              args.tokenizer, args.dicdir, args.mecabrc)

    for key in datadict:
        logger.info(
            f"data count for {key} split (before flatten): {len(datadict[key])}")

    # arrange data structure following the squad_v2 dataset in HF-hub
    def convert(example):
        qid = f'{example["qid"]}{example["doc_id"]:04d}'
        is_impossible = example["documents.score"] < 2
        if not is_impossible:
            answer = example["answer"]
            if args.tokenizer is not None:
                # search answer span in the tokenized context
                answer = "".join(ch for ch in answer if not ch.isspace())
                context = example["documents.text"]
                context_strip, offsets = zip(
                    *[(ch, ptr) for ptr, ch in enumerate(context) if not ch.isspace()])
                idx = "".join(context_strip).index(answer)
                answer_start, answer_end = offsets[idx], offsets[idx + len(
                    answer) - 1]
                answer = context[answer_start:answer_end + 1]
            else:
                answer_start = example["documents.text"].index(answer)

        return {
            "id": qid,
            "title": qid,
            "context": example["documents.text"],
            "question": example["question"],
            "answers": {"text": [answer] if not is_impossible else [],
                        "answer_start": [answer_start] if not is_impossible else [], },
        }
    datadict = datadict.map(
        convert, remove_columns=datadict["train"].column_names)

    for key in datadict:
        logger.info(
            f"data count for {key} split (after flatten): {len(datadict[key])}")

    return datadict


def normalize_texts(datadict: DatasetDict, text_columns: List[str], word_form: str):
    """normalize columns using sudachitra with given word_form"""
    if word_form == WordFormTypes.SURFACE:
        return datadict

    sudachi_dict = Dictionary(dict="core")
    sudachi = sudachi_dict.create(SplitMode.C)
    normalizer = InputStringNormalizer(do_lower_case=False, do_nfkc=True)
    formatter = word_formatter(word_form, sudachi_dict)

    def normalize(t): return "".join(
        formatter(m) for m in sudachi.tokenize(normalizer.normalize_str(t)))

    def convert(examples):
        for clm in text_columns:
            examples[clm] = [normalize(t) for t in examples[clm]]
        return examples

    datadict = datadict.map(convert, batched=True)
    return datadict


def tokenize_texts(datadict: DatasetDict, text_columns: List[str], tokenizer: str, dicdir=None, mecabrc=None) -> DatasetDict:
    """Tokenize texts with wakati-mode (outputs space delimited surfaces)."""

    if tokenizer is None:
        logger.info("keep texts as is.")
        return datadict
    elif tokenizer == "juman":
        logger.info("split text using Juman++")
        tok = tokenizer_utils.Juman()
    elif tokenizer == "mecab":
        logger.info(
            f"split text using MeCab (dicdir: {dicdir}, rc: {mecabrc})")
        tok = tokenizer_utils.MecabJuman(dicdir, mecabrc)
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    def convert(examples):
        for clm in text_columns:
            examples[clm] = [tok(t) for t in examples[clm]]
        return examples

    datadict = datadict.map(convert, batched=True)
    return datadict


def select_column(datadict: DatasetDict, rename_map: Dict[str, str]) -> DatasetDict:
    """Renames columns based on given dict and deletes other columns."""

    # assume there is train dataset
    column_names = datadict["train"].column_names

    datadict = datadict.remove_columns(
        [c for c in column_names if c not in rename_map])

    for k, v in rename_map.items():
        datadict = datadict.rename_column(k, v)

    return datadict


def shuffle_dataset(dataset: Dataset, seed=None) -> Dataset:
    """Shuffles dataset if seed is given."""
    if seed is not None:
        logger.info(f"shuffle data with seed {seed}.")
        return dataset.shuffle(seed=seed)
    else:
        logger.info(f"skip shuffle. set --seed option to shuffle.")
        return dataset


def split_dataset(dataset: Dataset, split_rate_str: str = None):
    """Splits dataset into train/dev/test set."""
    # ref: https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090

    DEFAULT_SPLIT_RATE = "8/1/1"
    split_rate_str = split_rate_str if split_rate_str is not None else DEFAULT_SPLIT_RATE

    v_train, v_val, v_test = (int(v) for v in split_rate_str.split("/"))
    if not (v_train > 0 and v_val >= 0 and v_test > 0):
        raise ValueError(f"invalid data split rate: {split_rate_str}. "
                         "train and test rate must be non-zero.")

    r_valtest = (v_val + v_test) / (v_train + v_val + v_test)
    r_test = v_test / (v_val + v_test)

    # train_test_split generates DatasetDict with name "train" and "test"
    train_testvalid = dataset.train_test_split(
        test_size=r_valtest, shuffle=False)

    if v_val == 0:
        datadict = DatasetDict({
            "train": train_testvalid["train"],
            "test": train_testvalid["test"],
        })
    else:
        test_valid = train_testvalid["test"].train_test_split(
            test_size=r_test, shuffle=False)

        datadict = DatasetDict({
            "train": train_testvalid["train"],
            "dev": test_valid["train"],
            "test": test_valid["test"],
        })

    logger.info(f"whole data count: {len(dataset)}")
    for key in datadict:
        logger.info(f"data count for {key} split: {len(datadict[key])}")

    return datadict


def construct_output_filepath(output_dir: Path, filename: str, output_format: str):
    return output_dir / f"{filename}.{output_format}"


def save_datasets(datadict, output_dir, output_format):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in ("train", "dev", "test"):
        if key not in datadict:
            continue

        out_file = construct_output_filepath(output_dir, key, output_format)
        if output_format == "csv":
            datadict[key].to_csv(out_file, index=False)
        elif output_format == "json":
            datadict[key].to_json(out_file)

    return


CONVERT_FUNCS = {
    "amazon": convert_amazon,
    "kuci": convert_kuci,
    "rcqa": convert_rcqa,
}
TOKENIZER_NAMES = ["juman", "mecab"]
OUTPUT_FORMATS = ["csv", "json"]


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument(dest="dataset_name", type=str,
                        help="Target dataset name. Set \"list\" to list available datasets.")

    parser.add_argument("-i", "--input", dest="dataset_dir_or_file", type=str,
                        help="Raw data file or directory contains it.")
    parser.add_argument("-o", "--output", dest="output_dir", type=str, default="./output",
                        help="Output directory.")
    parser.add_argument("-f", "--output-format", type=str, default="json",
                        help="Output data format. json or csv. json by default.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output files when they already exist.")
    parser.add_argument("--word-form", dest="word_form", type=str, default=str(WordFormTypes.SURFACE),
                        help=f"Word form type for sudachitra. If set, normalize text columns of data.")
    parser.add_argument("-t", "--tokenizer", type=str, default=None,
                        help=f"Tokenizer to split texts (wakati). Output raw texts if not set. "
                        f"One of {TOKENIZER_NAMES}.")
    parser.add_argument("--dicdir", type=str, default=None,
                        help="dicdir option for mecab tokenizer.")
    parser.add_argument("--mecabrc", type=str, default=None,
                        help="rcfile option for mecab tokenizer.")
    parser.add_argument("-s", "--seed", type=int,
                        help="Random seed for shuffle. SKIP shuffle if not set.")
    parser.add_argument("-r", "--split-rate", type=str,
                        help="Split rate for train/validation/test data. "
                        "By default use dataset specific split, 8/1/1 otherwise.")

    args = parser.parse_args()
    args.dataset_name = args.dataset_name.lower()
    if args.tokenizer is not None:
        args.tokenizer = args.tokenizer.lower()
    args.output_dir = Path(args.output_dir)
    args.output_format = args.output_format.lower()
    return args


def validate_args(args):
    if args.dataset_name not in CONVERT_FUNCS and args.dataset_name != "list":
        logger.error(f"Unknown dataset name ({args.dataset_name}). "
                     f"It must be one of {list(CONVERT_FUNCS.keys())} or \"list\".")
        raise ValueError

    if args.tokenizer is not None and args.tokenizer not in TOKENIZER_NAMES:
        logger.error(f"Unknown tokenizer ({args.tokenizer}). "
                     f"It must be one of {TOKENIZER_NAMES}.")
        raise ValueError

    if args.output_format not in OUTPUT_FORMATS:
        logger.error(f"Unknown dataset name ({args.output_format}). "
                     f"It must be one of {OUTPUT_FORMATS}.")
        raise ValueError

    if not args.overwrite:
        for key in ("train", "dev", "test"):
            out_file = construct_output_filepath(
                args.output_dir, key, args.output_format)
            if out_file.exists():
                logger.error(
                    f"File {out_file} already exists. Set --overwrite to continue anyway.")
                raise ValueError
    return


def main():
    args = parse_args()
    validate_args(args)

    if args.dataset_name == "list":
        logger.info(f"Available datasets: {list(CONVERT_FUNCS.keys())}")
        return

    convert_func = CONVERT_FUNCS[args.dataset_name]
    datadict = convert_func(args)
    save_datasets(datadict, args.output_dir, args.output_format)
    return


if __name__ == "__main__":
    main()
