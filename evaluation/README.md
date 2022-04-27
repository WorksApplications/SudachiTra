# Evaluation

This folder contains scripts to evaluate models.

# Evaluation methods

## Performance with downstream tasks

Evaluate model with 3 tasks.

### Tasks

- The Multilingual Amazon Reviews Corpus (Amazon)
  - https://registry.opendata.aws/amazon-reviews-ml/
  - Text classification task / 文章分類
- 京都大学常識推論データセット (KUCI)
  - https://nlp.ist.i.kyoto-u.ac.jp/?KUCI
  - Multiple choice task / 常識推論
- 解答可能性付き読解データセット (RCQA)
  - http://www.cl.ecei.tohoku.ac.jp/rcqa/
  - Question answering task (SQuAD2.0 format) / 読解

### Steps

Example for Amazon task (replace `amazon` with `kuci` or `rcqa` for other tasks):

```bash
# Generate dataset files for evaluation.
# Download raw data first for KUCI and RCQA.
python convert_dataset.py amazon --output ./datasets/amazon
# python convert_dataset.py kuci --input /path/to/input  --output ./datasets/kuci

# Run finetuning/prediction with hyper parameter search.
# `run_all.sh` runs with all 3 datasets and parameters.
# Place model files under `./bert/`.
./run_all.sh

# Correct test result file (for chitra surface model).
python summary_results.py amazon ./out/chitra_surface_amazon/ --output ./summary.csv
```

## Robustness to the text normalization

Run evaluation with test data whose texts are normalized.

Ideal model should be robust to this change (outputs remain same after nomralization).

### Steps

```bash
# Generate normalized dataset.
python convert_dataset.py amazon --output ./datasets_normalized/amazon

# Following steps are same to the model evaluation, but need to modify
# dataset dir name in `run_all.sh` to `datasets_normalized` in this case.
```

Rest steps are same.

# Script Usage

This section shows the list of scripts and their usage.
Also check the help of each scripts.

## install_jumanpp.sh

`install_jumanpp.sh` is a helper script to install Juman++.

Juman++ is neccessary to use `tokenizer_utils.Juman`, which will be used to tokenize data for Kyoto-U BERT.

The default install location is `$HOME/.local/usr`.
Modify the script as you want and set PATH.

## convert_dataset.py

`convert_dataset.py` is a script to preprocess datasets.
`run_evaluation.py` requires the dataset format produced by this script.

This script has `--seed` and `--split-rate` option to randomize train/dev/test set,
however, in our experiment we use default split (no option).

### example

```bash
# Amazon Review
# Raw data will be loaded from huggingface datasets hub.
python convert_dataset.py amazon -o ./amazon

# RCQA dataset
# Download raw data from http://www.cl.ecei.tohoku.ac.jp/rcqa/.
python convert_dataset.py rcqa -i ./all-v1.0.json.gz -o ./rcqa

# KUCI dataset
# Download raw data from https://nlp.ist.i.kyoto-u.ac.jp/?KUCI and untar.
python convert_dataset.py kuci -i ./KUCI/ -o ./kuci
```

### tokenize texts (wakati)

Some BERT models need texts in the dataset tokenized (wakati-gaki):

- NICT BERT: by MeCab (juman dic)
- Kyoto-U BERT: by Juman++

Use `--tokenize` option to tokenize text columns.
You need to install tokenizers to use.

Note that `run_evaluation.py` also has an option to tokenize text.

```bash
# tokenize with Juman++
python convert_dataset.py rcqa -i ./all-v1.0.json.gz --tokenize juman

# tokenize with MeCab (juman dic)
python convert_dataset.py rcqa -i ./all-v1.0.json.gz \
    --tokenize mecab --dicdir /var/lib/mecab/dic/juman-utf-8 --mecabrc /etc/mecabrc
```

### normalize texts

To check an impact of the normalization, we evaluate models with datasets whose texts are normalized.
`convert_dataset.py` has option for that.

Use `--word-form` option to apply sudachitra normalization to texts.
We used `normalized_and_surface` for our experiment.

```bash
python convert_dataset.py amazon --word-form normalized_and_surface
```

## run_evaluation.py

`run_evaluation.py` is a script to run a single evaluation (with single model, dataset, hyper-parameters).

Note:

- The model path for `--model_name_or_path` must contain `bert` to let `transformers.AutoModel` work correctly.
- To use sudachi tokenizer, set `sudachi` for `tokenizer_name`.
  - Script assumes that `vocab.txt` and `tokenizer_config.json` are in the model path.
- You may need to clear huggingface datasets cache file before running this script:
  - Dataset preprocessing will generate a cache file with random hash due to the our non-picklable conversion.
  - The random hash become same if you use same seed due to the set_seed.

### example

Template

```bash
python ./run_evaluation.py \
    --model_name_or_path          [./path/to/model or name in huggingface-hub] \
    --from_pt                     [set true if load pytorch model] \
    --pretokenizer_name           [set "juman" or "mecab-juman" to tokenize text before using HF-tokenizer] \
    --tokenizer_name              [set "sudachi" to use SudachiTokenizer] \
    --dataset_name                ["amazon" or "rcqa" or "kuci"] \
    --dataset_dir                 [./path/to/dataset/dir] \
    --output_dir                  [./path/to/output] \
    --do_train                    [set to finetune model] \
    --do_eval                     [set to evaluate model with dev set] \
    --do_predict                  [set to evaluate model with test data] \
    --per_device_eval_batch_size  [evaluation batch size] \
    --per_device_train_batch_size [training batch size] \
    --learning_rate               [learning rate] \
    --num_train_epochs            [epochs to finetune] \
    --overwrite_cache             [set to overwrite data preprocess cache] \
    --max_train_samples           [limit number of train samples (for test run)] \
    --max_val_samples             [limit number of val samples (for test run)] \
    --max_test_samples            [limit number of test samples (for test run)] \
```

Run finetuning with tohoku BERT and amazon dataset,
assuming dataset file (generated by `convert_dataset.py`) locates under `datasets/amazon/`.

```bash
python ./run_evaluation.py \
    --model_name_or_path          "cl-tohoku/bert-base-japanese-whole-word-masking" \
    --dataset_name                "amazon" \
    --dataset_dir                 ./datasets/amazon \
    --output_dir                  ./output/tohoku_amazon \
    --do_train                    \
    --per_device_eval_batch_size  64 \
    --per_device_train_batch_size 16 \
    --learning_rate               5e-5 \
    --num_train_epochs            4 \
    # --max_train_samples           100 \
    # --max_val_samples             100 \
    # --max_test_samples            100 \
```

Run prediction with NICT BERT and KUCI dataset.
Assume dataset is not tokenized.

```bash
python ./run_evaluation.py \
    --model_name_or_path          ./path/to/nict_bert/model \
    --pretokenizer_name           "mecab-juman" \
    --dataset_name                "kuci" \
    --dataset_dir                 ./datasets/kuci \
    --output_dir                  ./output/nict_kuci \
    --do_eval                     \
    --do_predict                  \
    --per_device_eval_batch_size  64 \
    --per_device_train_batch_size 16 \
    --learning_rate               5e-5 \
    --num_train_epochs            4 \
```

Run whole steps with chitra (normalized_and_surface) and RCQA dataset.

```bash
python ./run_evaluation.py \
    --model_name_or_path          ./path/to/chitra/model \
    --tokenizer_name              "sudachi" \
    --dataset_name                "rcqa" \
    --dataset_dir                 ./datasets/rcqa \
    --output_dir                  ./output/chitra_rcqa \
    --do_train                    \
    --do_eval                     \
    --do_predict                  \
    --per_device_eval_batch_size  64 \
    --per_device_train_batch_size 16 \
    --learning_rate               5e-5 \
    --num_train_epochs            4 \
```

## run_all.sh

`run_all.sh` is a script to run `run_evaluation.py` with different models and hyper parameters.

This assumes all model files are placed in the same directory (and named `bert` for `run_evaluation.py`), and all 3 datasets are placed in another directory.
You will need to set those directories in the script for your environment.

```bash
./run_all.sh
```

## summary_results.py

`summary_results.py` is a script to collect and summarize metrics of
test results of models with each hyper-parameters.

It requires the input directory has a structure generated by `run_all.sh`, i.e.:

```
input_dir
├── [hyper-parameter dirs (ex. "5e-5_32_3")]
│   ├── [validation result file]
│   └── [test result file]
...
```

```bash
python ./summary_results.py amazon -i ./out/chitra_amazon
```
