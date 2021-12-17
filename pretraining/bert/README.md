# Training Sudachi BERT Models

## Pretrained models

Pre-trained BERT models and tokenizer are coming soon!


## Set up

We use TensorFlow 2.x implementation for BERT to pretrain models.

https://github.com/tensorflow/models/tree/master/official/nlp/bert

To pretrain BERT models, you need to download [models](https://github.com/tensorflow/models) repository.

```shell script
$ git clone --recursive https://github.com/WorksApplications/SudachiTra/
```

In addition, you need to install the required packages to pretrain models.

```shell script
$ cd SudachiTra/
$ pip install -r pretraining/bert/requirements.txt
```

## Details of pretraining

The following sample codes are to train BERT models with [wiki40b](https://www.tensorflow.org/datasets/catalog/wiki40b) dataset.

### 1. Download Wiki40b

This script downloads [wiki40b](https://www.tensorflow.org/datasets/catalog/wiki40b) and split sentences for the data.
The default output dir is `./datasets`.

The sentence split data with the markup tags of article delimiter `_START_ARTICLE_` and paragraph delimiter `_START_PARAGRAPH_` will be stored in `./datasets/corpus/`.
The data divided into paragraphs as one document is saved in `./datasets/corpus_splitted_by_paragraph`.


```shell script
$ cd pretraining/bert/
# It may take several hours.
$ ./run_prepare_dataset.sh
```

### 2. Preprocessing: Corpus Cleaning

Some sentences in the downloaded corpus are too short or too long to have much impact on learning.
There are also documents that are too short or contain inappropriate words.
To filter out such sentences and documents, use the [`prepare_dataset.py`](preprocess_dataset.py) script.
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
$ python train_wordpiece_tokenizer.py \
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
$ python train_pos_substitution_tokenizer.py \
--input_file datasets/corpus_splitted_by_paragraph/ja_wiki40b_train.preprocessed.paragraph.txt \
--token_size 32000 \
--limit_character 5000 \
--dict_type core \
--split_mode C \
--word_form_type normalized \
--output_file _tokenizers/ja_wiki40b/pos_substitution/train_CoreDic_normalized_unit-C/vocab.txt 
```

### 4.Creating data for pretraining

We recommend to split dataset into multiple files.

```shell script
# splits wiki40b into multiple files
$ python split_dataset.py \
--input_file datasets/corpus_splitted_by_paragraph/ja_wiki40b_train.preprocessed.paragraph.txt \
--line_per_file 760000
```

```shell script
$ cd ../../
$ WORK_DIR="pretraining/bert"
$ seq -f %02g 1 8|xargs -L 1 -I {} -P 8 python3 $WORK_DIR/create_pretraining_data.py \
--input_file $WORK_DIR/datasets/corpus_splitted_by_paragraph/ja_wiki40b_train.preprocessed.paragraph{}.txt \
--output_file $WORK_DIR/bert/pretraining_train_{}.tf_record \
--do_nfkc \
--vocab_file $WORK_DIR/_tokenizers/ja_wiki40b/wordpiece/train_CoreDic_normalized_unit-C/wordpiece-vocab.txt \
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

```shell script
$ pwd
# /path/to/bert_sudachipy
$ sudo pip3 install -r models/official/requirements.txt
$ export PYTHONPATH="$PYTHONPATH:./models"
$ cd models/
$ WORK_DIR="../pretraining/bert"; py official/nlp/bert/run_pretraining.py \
--input_files="$WORK_DIR/bert/pretraining_*_*.tf_record" \
--model_dir="$WORK_DIR/bert_small/" \
--bert_config_file="$WORK_DIR/bert_small/bert_small_config.json" \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--train_batch_size=256 \
--learning_rate=1e-4 \
--num_train_epochs=100 \
--num_steps_per_epoch=10000 \
--optimizer_type=adamw \
--warmup_steps=10000
```

### 6.Converting a model to pytorch format

```shell script
$ cd ../pretraining/bert/
$ python convert_original_tf2_checkpoint_to_pytorch.py \
--tf_checkpoint_path ./bert_small/ \
--config_file ./bert_small/bert_small_config.json \
--pytorch_dump_path ./bert/bert_small/pytorch_model.bin
```