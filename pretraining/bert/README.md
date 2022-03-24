# Training Sudachi BERT Models

This repository also provides a script and recipe to train the [BERT](https://arxiv.org/abs/1810.04805) model.  

## Pretrained models

You can download the pretrained models from [README](../../README.md).

## Set up

In order to pretrain models, you need to download this repository, including its submodules.

```shell script
$ git clone --recursive https://github.com/WorksApplications/SudachiTra/
```

In addition, you need to install the required packages to pretrain models.

```shell script
$ pip install -U sudachitra
$ cd SudachiTra/
$ pip install -r requirements.txt
$ pip install -r pretraining/bert/requirements.txt
$ pip install -r pretraining/bert/models/official/requirements.txt
```

## Quick Start Guide

In the following guide, we use [wiki40b](https://www.tensorflow.org/datasets/catalog/wiki40b) dataset.

### 1. Download Wiki40b

For pre-training BERT, you need to prepare the data split into document units.

The [`run_prepare_dataset.sh`](run_prepare_dataset.sh) script launches download and processing of wiki40b.
The component steps in the script to prepare the datasets are as follows:

* Data download - wiki40b is downloaded in the `datasets/corpus` directory.
* Sentence segmentation - the corpus text is processed into separate sentences.
* Document segmentation - the corpus text divided into document.

The processed data is saved in `./datasets/corpus_splitted_by_paragraph`.
The corpus files are approximately 4.0GB in total.

```shell script
$ cd pretraining/bert/
# It may take several hours.
$ ./run_prepare_dataset.sh
```

### 2. Preprocessing: Corpus Cleaning

Some sentences in the downloaded corpus are too short or too long to have much impact on learning.
There are also documents that are too short or contain inappropriate words.
To filter out such sentences and documents, use the [`preprocess_dataset.py`](preprocess_dataset.py) script.
In addition to the cleaning process, this script also performs the sentence-level and document-level normalization process.

Example script to apply all cleaning and normalization processes.

```shell
$ py preprocess_dataset.py  \
-i ./datasets/corpus_splitted_by_paragraph/ja_wiki40b_train.paragraph.txt \
-o ./datasets/corpus_splitted_by_paragraph/ja_wiki40b_train.preprocessed.paragraph.txt \
--sentence_filter_names email url sequence_length \
--document_filter_names short_document script ng_words \
--sentence_normalizer_names citation whitespace \
--document_normalizer_names concat_short_sentence
```


### 3. Building vocabulary

You can specify tokenizer options of SudachiPy, such as sudachi dictionaries, split modes, and word forms.

The following word forms are available:

* `surface`
* `dictionary`
* `normalized`
* `dictionary_and_surface`
* `normalized_and_surface`

A implements three kinds of subword tokenizers:

* `WordPiece`
* `Character`
* `POS Substitution (part-of-speech substitution)`

#### WordPiece

We used WordPiece to obtain subwords.
We used an implementation of WordPiece in [Tokenizers](https://github.com/huggingface/tokenizers).

```shell script
$ python3 train_wordpiece_tokenizer.py \
--input_file datasets/corpus_splitted_by_paragraph/ja_wiki40b_train.preprocessed.paragraph.txt \
--do_nfkc \
--vocab_size 32000 \
--limit_alphabet 5000 \
--dict_type core \
--split_mode C \
--word_form_type normalized \
--output_dir _tokenizers/ja_wiki40b/wordpiece/train_CoreDic_normalized_unit-C \
--config_name config.json \
--vocab_prefix wordpiece
```

#### Character

You can get a vocabulary for `Character` by extracting only the characters from the vocabulary created by `Wordpiece` tokenization.

```shell script
# e.g. #characters(5,000) + #special_tokens(5) = 5,005
$ OUTPUT_DIR="tokenizers/ja_wiki40b/character/train_CoreDic_normalized_unit-C"
$ mkdir -p $OUTPUT_DIR
$ head -n 5005 _tokenizers/ja_wiki40b/wordpiece/train_CoreDic_normalized_unit-C/wordpiece-vocab.txt > $OUTPUT_DIR/vocab.txt
```

#### POS Substitution (part-of-speech substitution)

`POS Substitution` is a method using part-of-speech tags to reduce a vocabulary size.
In `POS Substitution`, instead of using a subword tokenizer, low frequency words are replaced by part-of-speech tags.
Finally, only part-of-speech tags that do not appear in a training corpus are treated as unknown words.


```shell script
$ python3 train_pos_substitution_tokenizer.py \
--input_file datasets/corpus_splitted_by_paragraph/ja_wiki40b_train.preprocessed.paragraph.txt \
--token_size 32000 \
--limit_character 5000 \
--dict_type core \
--split_mode C \
--word_form_type normalized \
--output_file _tokenizers/ja_wiki40b/pos_substitution/train_CoreDic_normalized_unit-C/vocab.txt 
```

### 4.Creating data for pretraining

To create the data for pre-training, we utilize a code based on [TensorFlow Model Garden](https://github.com/tensorflow/models).
The code to create the pre-training data with the tokenizer modified for SudachiPy is `pretraining/models/official/nlp/data/create_pretraining_data.py`.

This code will consume a lot of memory.
We can handle this by splitting the training corpus into multiple files and processing them in parallel.
Therefore, we recommend split train data into multiple files.

In the following example, the number of sentences per file (`--line_per_file`) is set to 700,000.
It consumes about 10 GB or more of memory to create the data for pre-training from this one file.


```shell script
# splits wiki40b into multiple files
$ python3 split_dataset.py \
--input_file datasets/corpus_splitted_by_paragraph/ja_wiki40b_train.preprocessed.paragraph.txt \
--line_per_file 700000
$ TRAIN_FILE_NUM=`find datasets/corpus_splitted_by_paragraph -type f | grep -E "ja_wiki40b_train.preprocessed.paragraph[0-9]+.txt" | wc -l`
```

```shell script
# Change the value according to the execution environment.
$ MAX_PROCS=8

$ mkdir datasets_for_pretraining
$ export $PYTHONPATH="$PYTHONPATH:./models"
$ seq 1 ${TRAIN_FILE_NUM} | xargs -L 1 -I {} -P ${MAX_PROCS} python3 models/official/nlp/data/create_pretraining_data.py \
--input_file datasets/corpus_splitted_by_paragraph/ja_wiki40b_train.preprocessed.paragraph{}.txt \
--output_file datasets_for_pretraining/pretraining_train_{}.tf_record \
--do_nfkc \
--vocab_file _tokenizers/ja_wiki40b/wordpiece/train_CoreDic_normalized_unit-C/wordpiece-vocab.txt \
--tokenizer_type wordpiece \
--word_form_type normalized \
--split_mode C \
--sudachi_dic_type core \
--do_whole_word_mask \
--max_seq_length 512 \
--max_predictions_per_seq 80 \
--dupe_factor 10
```

### 5.Training

#### NVIDIA DeepLearningExamples

To pretrain a model, we utilize a code based on [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples).

nvidia-docker is used.
Put the train data in this directory (`SudachiTra/pretraining/bert/DeepLearningExamples/TensorFlow2/LanguageModeling/BERT/data`).

```shell script
$ docker pull nvcr.io/nvidia/tensorflow:21.10-tf2-py3
$ cd DeepLearningExamples/TensorFlow2/LanguageModeling/BERT
$ bash scripts/docker/build.sh
$ bash scripts/docker/launch.sh

$ python3 data/bertPrep.py --action download --dataset google_pretrained_weights # Change the config if necessary.  ex. vocab_size
$ bash scripts/run_pretraining_lamb.sh 176 22 8 7.5e-4 5e-4 tf32 true 4 2000 200 11374 100 64 192 base # Change the path in run_pretraining_lamb if necessary.
```

### 6.Converting a model to pytorch format

#### NVIDIA DeepLearningExamples

To convert the generated model checkpoints to Pytorch, you can use `convert_original_tf2_checkpoint_to_pytorch_nvidia.py`.

```shell script
$ cd SudachiTra/pretraining/bert/
$ python3 convert_original_tf2_checkpoint_to_pytorch_nvidia.py \
--tf_checkpoint_path /path/to/checkpoint \
--config_file /path/to/bert_config.json \
--pytorch_dump_path /path/to/pytorch_model.bin
```