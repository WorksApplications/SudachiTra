# Training Sudachi BERT Models

## Pretrained models

Pre-trained BERT models and tokenizer are coming soon!


## Set up

We use TensorFlow 2.x implementation for BERT to pretrain models.

https://github.com/tensorflow/models/tree/master/official/nlp/bert

To pretrain BERT models, you need to download [models](https://github.com/tensorflow/models) repository.

```shell script
$ git clone --recursive https://github.com/WorksApplications/chitra/
```

In addition, you need to install the required packages to pretrain models.

```shell script
$ cd chitra/
$ pip install -r pretraining/bert/requirements.txt
```

## Details of pretraining

### 1. Corpus generation and preprocessing

The following sample codes are to train BERT models with [wiki40b](https://www.tensorflow.org/datasets/catalog/wiki40b) dataset.

This script downloads [wiki40b](https://www.tensorflow.org/datasets/catalog/wiki40b) onto `./datasets/corpus` and prepares paragraph-splitted data in ``./datasets/corpus_splitted_by_paragraph`.


```shell script
$ cd pretraining/bert/
# It may take several hours.
$ ./run_prepare_dataset.sh
```


### 2. Building vocabulary

You can specify tokenizer options of SudachiPy, such as sudachi dictionaries, split modes, and word forms.

The following word forms are available:

* `surface`
* `dictionary`
* `normalized`
* `dictionary_and_surface`
* `normalized_and_surface`

We used WordPiece to obtain subwords.
We used an implementation of WordPiece in [Tokenizers](https://github.com/huggingface/tokenizers).

```shell script
$ python train_pretokenizer.py \
--input_file datasets/corpus_splitted_by_paragraph/ja_wiki40b_*.paragraph.txt \
--dict_type core
--split_mode C
--word_form_type normalized
--output_dir ja_wiki40b_*_CoreDic_normalized_unit-C
--config_name ja_wiki40b_*_CoreDic_normalized_unit-C_config.json
--vocab_prefix ja_wiki40b_*_CoreDic_normalized_unit-C
```

### 3.Creating data for pretraining

First, you need to split dataset into multiple files.

```shell script
# splits wiki40b into multiple files
$ python data_split.py \
--input_file datasets/corpus_splitted_by_paragraph/ja_wiki40b_*.paragraph.txt \
--line_per_file 760000
```

Please refer to `./run_create_pretraining_data.sh` to create pretraining data.

### 4.Training

```shell script
$ pwd
# /path/to/bert_sudachipy
$ sudo pip3 install -r models/official/requirements.txt
$ export PYTHONPATH="$PYTHONPATH:./models"
$ cd models/
$ WORK_DIR="../pretraining/bert"; py official/nlp/bert/run_pretraining.py \
--input_files="$WORK_DIR/models/pretraining_small_*record" \
--model_dir="$WORK_DIR/models/" \
--bert_config_file="$WORK_DIR/models/small_config.json" \
--max_seq_length=512 \
--max_predictions_per_seq=80 \
--train_batch_size=256 \
--learning_rate=1e-4 \
--num_train_epochs=100 \
--num_steps_per_epoch=10000 \
--optimizer_type=adamw \
--warmup_steps=10000
```

### 5.Converting a model to pytorch format

```shell script
cd ../pretraining/bert/
python convert_original_tf2_checkpoint_to_pytorch.py \
--tf_checkpoint_path ./models/ \
--config_file ./models/small_config.json \
--pytorch_dump_path ./models/pytorch_model.bin
```