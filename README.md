# chiTra

[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![test](https://github.com/t-yamamura/bert_sudachipy/actions/workflows/test.yaml/badge.svg)](https://github.com/t-yamamura/bert_sudachipy/actions/workflows/test.yaml)
[![](https://img.shields.io/github/license/t-yamamura/bert_sudachipy.svg)](https://github.com/t-yamamura/bert_sudachipy/blob/main/LICENSE)

chiTra is a Japanese tokenizer for [Transformers](https://github.com/huggingface/transformers).

chiTra stands for Suda**chi** for **Tra**nsformers.


## Quick Tour

```python
>>> from transformers import BertModel
>>> from chitra.tokenization_bert_sudachipy import BertSudachipyTokenizer

>>> tokenizer = BertSudachipyTokenizer.from_pretrained('chitra-bert-base-japanese-sudachi')
>>> model = BertModel.from_pretrained('chitra-bert-base-japanese-sudachi')
>>> model(**tokenizer("まさにオールマイティーな商品だ。", return_tensors="pt")).last_hidden_state
```

> Pre-trained BERT models and tokenizer are coming soon!


## Installation

```shell script
$ pip install chitra
```

The default [Sudachi dictionary](https://github.com/WorksApplications/SudachiDict) is [SudachiDict-core](https://pypi.org/project/SudachiDict-core/).
You can use other dictionaries, such as [SudachiDict-small](https://pypi.org/project/SudachiDict-small/) and [SudachiDict-full](https://pypi.org/project/SudachiDict-full/).
In such cases, you need to install the dictionaries.

```shell script
$ pip install sudachidict_small sudachidict_full
```


## Pretraining

Please refer to [pretraining/bert/README.md](https://github.com/t-yamamura/bert_sudachipy/tree/main/pretraining/bert).


## Roadmap

* Releasing pre-trained models for BERT


## For Developers

TBD


## Contact

chiTra are developed by [WAP Tokushima Laboratory of AI and NLP](http://nlp.worksap.co.jp/).

Open an issue, or come to our Slack workspace for questions and discussion.

https://sudachi-dev.slack.com/ (Get invitation [here](https://join.slack.com/t/sudachi-dev/shared_invite/enQtMzg2NTI2NjYxNTUyLTMyYmNkZWQ0Y2E5NmQxMTI3ZGM3NDU0NzU4NGE1Y2UwYTVmNTViYjJmNDI0MWZiYTg4ODNmMzgxYTQ3ZmI2OWU))

Enjoy tokenization!
