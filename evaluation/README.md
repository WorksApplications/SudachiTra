This directory contains scripts to evaluate BERT models.

# Script Usage

## install_jumanpp.sh

`install_jumanpp.sh` is a shell script to install Juman++.

Juman++ is neccessary to use `tokenizer_util.Juman`, which will be used to tokenize data for Kyoto-U BERT.


## convert_dataset.py

`convert_dataset.py` is a script to preprocess datasets.
`evaluate_model.py` assumes the dataset preprocessed by this.

### example

```bash
# Amazon Review
# will be loaded from huggingface datasets hub
python convert_dataset.py amazon -o ./amazon

# RCQA dataset
# download raw data first
python convert_dataset.py rcqa -i ./all-v1.0.json.gz -o ./rcqa

# KUCI dataset
# download raw data and untar first
python convert_dataset.py kuci -i ./KUCI/ -o ./kuci
```

Some BERT models need texts in the dataset tokenized:

- NICT BERT: by MeCab (juman dic)
- Kyoto-U BERT: by Juman++

Note that `evaluate_model.py` also has an option to tokenize text.

```bash
# tokenize with Juman++
python convert_dataset.py rcqa -i ./all-v1.0.json.gz --tokenize juman

# tokenize with MeCab (juman dic)
python convert_dataset.py rcqa -i ./all-v1.0.json.gz \
    --tokenize mecab --dicdir /var/lib/mecab/dic/juman-utf-8 --mecabrc /etc/mecabrc
```


## evaluate_model.py

`evaluate_model.py` is a script for a single evaluation run.

### example

Template

```bash
python ./evaluate_model.py \
    --model_name_or_path          [./path/to/model or name in huggingface-hub] \
    --from_pt                     [set true if load pytorch model] \
    --pretokenizer_name           [set "juman" or "mecab-juman" to tokenize text before using HF-tokenizer] \
    --tokenizer_name              [set "sudachi" to use SudachiTokenizer] \
    --word_form_type              [arg for sudachi tokenizer] \
    --split_unit_type             [arg for sudachi tokenizer] \
    --sudachi_vocab_file          [arg for sudachi tokenizer] \
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
    --max_train_samples           [limit number of train samples (for test run)] \
    --max_val_samples             [limit number of val samples (for test run)] \
    --max_test_samples            [limit number of test samples (for test run)] \
```

Run finetuning with tohoku-bert and amazon dataset

```
python ./evaluate_model.py \
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


## finetune_all_models.sh

`finetune_all_models.sh` is a script to run `evaluate_model.py` with different parameters.

Before running this script, you need to modify some variales in it.

```bash
./finetune_all_models.sh amazon
```

