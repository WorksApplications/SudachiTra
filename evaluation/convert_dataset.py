import argparse as ap
import logging
import sys
from pathlib import Path

from datasets import load_dataset, DatasetDict


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


# convert funcs
def convert_amazon(args):
    dataset_file = Path(args.dataset_dir_or_file)
    FILE_NAME = "amazon_reviews_multilingual_JP_v1_00.tsv"
    if dataset_file.is_dir():
        dataset_file = dataset_file / FILE_NAME
    if not dataset_file.exists() or dataset_file.name != FILE_NAME:
        raise ValueError(f"file {FILE_NAME} does not found in dataset_dir. "
                         "Download and unzip it first (https://s3.amazonaws.com/amazon-reviews-pds/readme.html).")

    dataset = load_dataset("csv", data_files=str(
        dataset_file), delimiter='\t')["train"]

    dataset = select_column(dataset, {
        "review_body": "sentence1",
        "star_rating": "label",
    })

    # shuffle if seed is given
    if args.seed is not None:
        logger.info(f"shuffle data with seed {args.seed}")
        dataset = dataset.shuffle(seed=args.seed)

    dataset = split_dataset(dataset, args.split_rate)

    return dataset


def convert_livedoor(args):
    # todo
    dataset_dir = Path(args.dataset_dir_or_file)
    return


# util funcs
def select_column(dataset, rename_map):
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in rename_map])

    for k, v in rename_map.items():
        dataset = dataset.rename_column(k, v)

    return dataset


def split_dataset(dataset, split_rate_str):
    logger.info(f"whole data count: {len(dataset)}")
    # split into 3 parts
    # ref: https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090
    v_train, v_val, v_test = (int(v) for v in split_rate_str.split("/"))

    if not (v_train > 0 and v_val >= 0 and v_test > 0):
        raise ValueError(f"invalid data split rate: {split_rate_str}. "
                         "train and test rate must be non-zero.")

    r_valtest = (v_val + v_test) / (v_train + v_val + v_test)
    r_test = v_test / (v_val + v_test)

    train_testvalid = dataset.train_test_split(
        test_size=r_valtest, shuffle=False)

    if v_val == 0:
        dataset = DatasetDict({
            "train": train_testvalid["train"],
            "test": train_testvalid["test"],
        })
    else:
        test_valid = train_testvalid["test"].train_test_split(
            test_size=r_test, shuffle=False)

        dataset = DatasetDict({
            "train": train_testvalid["train"],
            "dev": test_valid["train"],
            "test": test_valid["test"],
        })

    for key in dataset:
        logger.info(f"data count for {key} split: {len(dataset[key])}")

    return dataset


def save_dataset(dataset, output_dir, output_format):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in ("train", "dev", "test"):
        if key not in dataset:
            continue

        out_file = output_dir / f"{key}.{output_format}"

        if output_format == "csv":
            dataset[key].to_csv(out_file, index=False)
        elif output_format == "json":
            dataset[key].to_json(out_file)

    return


DATASET_FUNCS = {"amazon": convert_amazon}
OUTPUT_FORMATS = ["csv", "json"]


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument(dest="dataset_name", type=str,
                        help="target dataset name")
    parser.add_argument(dest="dataset_dir_or_file", type=str,
                        help="directory where the raw data locate")
    parser.add_argument(dest="output_dir", type=str, help="output directory")

    parser.add_argument('-s', "--seed", type=int,
                        default=None, help="random seed")
    parser.add_argument('-f', "--output-format", type=str,
                        default="csv", help="output format")
    parser.add_argument("-r", "--split-rate", type=str,
                        default="8/1/1", help="split rate for train/validation/test data. 8/1/1 by default.")

    args = parser.parse_args()

    # check
    if args.dataset_name not in DATASET_FUNCS:
        raise ValueError(f"unknown dataset name ({args.dataset_name}). "
                         f"It must be one of {list(DATASET_FUNCS.keys())}")
    if args.output_format not in OUTPUT_FORMATS:
        raise ValueError(f"unknown dataset name ({args.output_format}). "
                         f"It must be one of {OUTPUT_FORMATS}")

    return args


def main():
    args = parse_args()

    convert_func = DATASET_FUNCS[args.dataset_name]
    dataset = convert_func(args)

    save_dataset(dataset, args.output_dir, args.output_format)
    return


if __name__ == "__main__":
    main()
