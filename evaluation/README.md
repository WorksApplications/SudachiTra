# Evaluation

This folder contains scripts to evaluate models.

# Evaluation methods

## Performance on downstream tasks

We evaluated chiTra models with below 3 tasks.

- [The Multilingual Amazon Reviews Corpus (Amazon)](https://registry.opendata.aws/amazon-reviews-ml/)
  - Text classification task / 文章分類
  - We used 'ja' subset only.
  - We used `review_body` column as an input text and `stars` column (1 to 5) as a target class.
- [京都大学常識推論データセット (KUCI)](https://nlp.ist.i.kyoto-u.ac.jp/?KUCI)
  - Multiple choice task / 常識推論
- [解答可能性付き読解データセット (RCQA)](http://www.cl.ecei.tohoku.ac.jp/rcqa/)
  - Question answering task (SQuAD 2.0 format) / 読解

### Steps

0. Framework

We prepare evaluation scripts with `pytorch` and `tensorflow`.
The scripts in those directories work equivalently.

We prepared those scripts with reference to example code of transformers, before [v4.16.0](https://github.com/huggingface/transformers/tree/v4.16.0).
We confirmed they works with transformers-v4.26.1, but you may need updates.

1. Prepare datasets

Use `convert_dataset.py` to convert datsets into suitable format.
For KUCI and RCQA task, you need to download original data beforehand.

```bash
python convert_dataset.py amazon --output /datasets/amazon
python convert_dataset.py kuci --input /datasets/KUCI --output /datasets/kuci
python convert_dataset.py rcqa --input /datasets/all-v1.0.json.gz --output /datasets/rcqa
```

By default, we assumes all 3 datasets locate at the same directory.
We also assume the directory name of each datasets are: `amazon`, `kuci`, `rcqa` (case sensitive).

2. Modify script

You need to modify `run_all.sh` script to set pathes to datasets and models.

- IO
  - `SCRIPT_DIR`: Path to the `SudachiTra/evaluation/pytorch` or `tensorflow`.
  - `OUTPUT_ROOT`: Path to the directory where experiment results will be written.
- Datasets
  - `DATASET_ROOT`: Path to the directory where you prepared datasets.
  - `DATASETS`: List of tasks to evaluate. Used for the output directory name.
- Models
  - `MODEL_ROOT`: Path to the directory where you put models to evaluate.
    - Not neccessary if you set `MODEL_DIRS` by yourself.
  - `MODEL_NAMES`: List of models to evaluate. Used for the output directory name.
  - `MODEL_DIRS`: Mapping from model name to the model directory or huggingface model name.

Note: You need to include `bert` in the model path to automaticaly load BERT models.

3. Run and collect results.

Use modified `run_all.sh` to run evaluation with each models, tasks and hyper parameters.

```bash
# Install lib
python -m pip install -U -r /path/to/SudachiTra/evaluation/pytorch/requirements.txt
# You need additional libraries to use touhoku-bert:
# python -m pip install fugashi ipadic unidic_lite

# Run experiment
/path/to/SudachiTra/evaluation/pytorch/run_all.sh
```

`run_all.sh` will write outputs to directories named `[model_name]_[task_name]/[learning_rate]_[batch_size]_[num_epoch]` under the `OUTPUT_ROOT`.
Use `summary_results.py` to gather result files of each tasks.

```bash
# Correct test result file.
python summary_results.py amazon /output/chitra_amazon/ --output /summary/amazon.csv
python summary_results.py kuci /output/chitra_kuci/ --output /summary/kuci.csv
python summary_results.py rcqa /output/chitra_rcqa/ --output /summary/rcqa.csv
```

## Robustness to the text normalization

Run evaluation with test data whose texts are normalized.

Ideal model should be robust to this change (outputs remain same after nomralization).

### Steps

Prepare text normalized datasets using `convert_dataset.py` with `--word-form` option.
Provide chiTra tokenizer `word_form_type` to specify the normalization type.
We used `normalized_and_surface` for our experiment.

```bash
python convert_dataset.py amazon --output /datasets_normalized/amazon \
    --word-form normalized_and_surface
```

Run experiments in the same way with modified datasets (see above section).

# Results

TBA

Also check [our paper](https://github.com/WorksApplications/SudachiTra#chitra%E3%81%AE%E5%BC%95%E7%94%A8--citing-chitra).

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

### Example

```bash
# Amazon Review
# Raw data will be loaded from huggingface datasets hub.
python convert_dataset.py amazon --output ./amazon

# RCQA dataset
# Download raw data from http://www.cl.ecei.tohoku.ac.jp/rcqa/ beforehand.
python convert_dataset.py rcqa --input ./all-v1.0.json.gz --output ./rcqa

# KUCI dataset
# Download raw data from https://nlp.ist.i.kyoto-u.ac.jp/?KUCI and untar beforehand.
python convert_dataset.py kuci --input ./KUCI/ --output ./kuci
```

### Tokenize texts (wakati)

Some BERT models need texts in the dataset tokenized (wakati-gaki):

- NICT BERT: by MeCab (juman dic)
- Kyoto-U BERT: by Juman++

Use `--tokenize` option to tokenize text columns.
You need to install tokenizers to use.

Note that `run_evaluation.py` also has an option to tokenize text.

```bash
# tokenize with Juman++
python convert_dataset.py rcqa \
    --input ./all-v1.0.json.gz --output ./rcqa_juman \
    --tokenize juman

# tokenize with MeCab (juman dic)
python convert_dataset.py rcqa \
    --input ./all-v1.0.json.gz --output ./rcqa_mecab \
    --tokenize mecab --dicdir /var/lib/mecab/dic/juman-utf-8 --mecabrc /etc/mecabrc
```

### Normalize texts

You can use `convert_dataset.py` to generate datasets for [testing model robustness](#robustness-to-the-text-normalization).

Provide `word_form_type` using `--word-form` option to apply sudachitra normalization to texts in datasets.
We used `normalized_and_surface` for our experiment.

```bash
python convert_dataset.py amazon --output ./amazon_normalized \
    --word-form normalized_and_surface
```

## run_evaluation.py

`run_evaluation.py` is a script to run a single evaluation (with single model, dataset, set of hyper parameters).

Note:

- The model path for `--model_name_or_path` must contain `bert` to let `transformers.AutoModel` work correctly.
- To use sudachi tokenizer, set `sudachi` for `tokenizer_name`.
  - Script assumes that `vocab.txt` and `tokenizer_config.json` are in the model path.
- You may need to clear huggingface datasets cache file before running this script:
  - Dataset preprocessing will generate a cache file with random hash due to the our non-picklable conversion.
  - The random hash become same if you use same seed due to the set_seed.

### Example

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

`run_all.sh` is a script to run `run_evaluation.py` with different models, tasks and hyper parameters.

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
[model and task dir (e.g. "chitra_amazon")]
├── [hyper-parameter dirs (ex. "5e-5_32_3")]
│   ├── [validation result file]
│   └── [test result file]
...
```

```bash
python ./summary_results.py amazon -i ./out/chitra_amazon
```
