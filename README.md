# Sudachi Transformers (chiTra)

[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![test](https://github.com/WorksApplications/SudachiTra/actions/workflows/test.yaml/badge.svg)](https://github.com/WorksApplications/SudachiTra/actions/workflows/test.yaml)
[![](https://img.shields.io/github/license/WorksApplications/SudachiTra.svg)](https://github.com/WorksApplications/SudachiTra/blob/main/LICENSE)

chiTraは事前学習済みの大規模な言語モデルと [Transformers](https://github.com/huggingface/transformers) 向けの日本語形態素解析器を提供します。 / chiTra provides the pre-trained language models and a Japanese tokenizer for [Transformers](https://github.com/huggingface/transformers).

chiTraはSuda**chi Tra**nsformersの略称です。 / chiTra stands for Suda**chi Tra**nsformers.

## 事前学習済みモデル / Pretrained Model
公開データは [Open Data Sponsorship Program](https://registry.opendata.aws/sudachi/) を使用してAWSでホストされています。 / Datas are generously hosted by AWS with their [Open Data Sponsorship Program](https://registry.opendata.aws/sudachi/).

| Version | Normalized             | SudachiTra | Sudachi | SudachiDict   | Text         | Pretrained Model                                                                            |
| ------- | ---------------------- | ---------- | ------- | ------------- | ------------ | ------------------------------------------------------------------------------------------- |
| v1.0    | normalized_and_surface | v0.1.7     | 0.6.2   | 20211220-core | NWJC (109GB) | 395 MB ([tar.gz](https://sudachi.s3.ap-northeast-1.amazonaws.com/chitra/chiTra-1.0.tar.gz)) | 
| v1.1    | normalized_nouns       | v0.1.8     | 0.6.6   | 20220729-core | NWJC with additional cleaning (79GB) | 396 MB ([tar.gz](https://sudachi.s3.ap-northeast-1.amazonaws.com/chitra/chiTra-1.1.tar.gz)) |

### 特長 / Features
- 大規模テキストによる学習 / Training on large texts
  - 国語研日本語ウェブコーパス (NWJC) をつかってモデルを学習することで多様な表現とさまざまなドメインに対応しています /  Models are trained on NINJAL Web Japanese Corpus (NWJC) to support a wide variety of expressions and domains.
- Sudachi の利用 / Using Sudachi
  - 形態素解析器 Sudachi を利用することで表記ゆれによる弊害を抑えています / By using the morphological analyzer Sudachi, reduce the negative effects of various notations.

# chiTraの使い方 / How to use chiTra

## クイックツアー / Quick Tour
事前準備 / Requirements
```bash
$ pip install sudachitra
$ wget https://sudachi.s3.ap-northeast-1.amazonaws.com/chitra/chiTra-1.1.tar.gz
$ tar -zxvf chiTra-1.1.tar.gz
```

モデルの読み込み / Load the model
```python
>>> from sudachitra.tokenization_bert_sudachipy import BertSudachipyTokenizer
>>> from transformers import BertModel

>>> tokenizer = BertSudachipyTokenizer.from_pretrained('chiTra-1.1')
>>> tokenizer.tokenize("選挙管理委員会とすだち")
['選挙', '##管理', '##委員会', 'と', '酢', '##橘']

>>> model = BertModel.from_pretrained('chiTra-1.1')
>>> model(**tokenizer("まさにオールマイティーな商品だ。", return_tensors="pt")).last_hidden_state
tensor([[[ 0.8583, -1.1752, -0.7987,  ..., -1.1691, -0.8355,  3.4678],
         [ 0.0220,  1.1702, -2.3334,  ...,  0.6673, -2.0774,  2.7731],
         [ 0.0894, -1.3009,  3.4650,  ..., -0.1140,  0.1767,  1.9859],
         ...,
         [-0.4429, -1.6267, -2.1493,  ..., -1.7801, -1.8009,  2.5343],
         [ 1.7204, -1.0540, -0.4362,  ..., -0.0228,  0.5622,  2.5800],
         [ 1.1125, -0.3986,  1.8532,  ..., -0.8021, -1.5888,  2.9520]]],
       grad_fn=<NativeLayerNormBackward0>)
```

## インストール / Installation

```shell script
$ pip install sudachitra
```

デフォルトの [Sudachi dictionary](https://github.com/WorksApplications/SudachiDict) は [SudachiDict-core](https://pypi.org/project/SudachiDict-core/) を使用します。 / The default [Sudachi dictionary](https://github.com/WorksApplications/SudachiDict) is [SudachiDict-core](https://pypi.org/project/SudachiDict-core/).

[SudachiDict-small](https://pypi.org/project/SudachiDict-small/) や [SudachiDict-full](https://pypi.org/project/SudachiDict-full/) など他の辞書をインストールして使用することもできます。 / You can use other dictionaries, such as [SudachiDict-small](https://pypi.org/project/SudachiDict-small/) and [SudachiDict-full](https://pypi.org/project/SudachiDict-full/) .<br/>
その場合は以下のように使いたい辞書をインストールしてください。 / In such cases, you need to install the dictionaries.<br/>
事前学習済みモデルを使いたい場合はcore辞書を使用して学習されていることに注意してください。 / If you want to use a pre-trained model, note that it is trained with SudachiDict-core.

```shell script
$ pip install sudachidict_small sudachidict_full
```

## 事前学習 / Pretraining

事前学習方法の詳細は [pretraining/bert/README.md](https://github.com/WorksApplications/SudachiTra/tree/main/pretraining/bert) を参照ください。 / Please refer to [pretraining/bert/README.md](https://github.com/WorksApplications/SudachiTra/tree/main/pretraining/bert).


## 開発者向け / For Developers
TBD

## ライセンス / License

Copyright (c) 2022 National Institute for Japanese Language and Linguistics and Works Applications Co., Ltd. All rights reserved.

"chiTra"は [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) で [国立国語研究所](https://www.ninjal.ac.jp/) 及び [株式会社ワークスアプリケーションズ](https://www.worksap.co.jp/) によって提供されています。 / "chiTra" is distributed by [National Institute for Japanese Language and Linguistics](https://www.ninjal.ac.jp/) and [Works Applications Co.,Ltd.](https://www.worksap.co.jp/) under [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).


## 連絡先 / Contact us
質問があれば、issueやslackをご利用ください。 / Open an issue, or come to our Slack workspace for questions and discussion.

開発者やユーザーの方々が質問したり議論するためのSlackワークスペースを用意しています。 / We have a Slack workspace for developers and users to ask questions and discuss.
https://sudachi-dev.slack.com/ ( [こちら](https://join.slack.com/t/sudachi-dev/shared_invite/enQtMzg2NTI2NjYxNTUyLTMyYmNkZWQ0Y2E5NmQxMTI3ZGM3NDU0NzU4NGE1Y2UwYTVmNTViYjJmNDI0MWZiYTg4ODNmMzgxYTQ3ZmI2OWU) から招待を受けてください) / https://sudachi-dev.slack.com/ (Get invitation [here](https://join.slack.com/t/sudachi-dev/shared_invite/enQtMzg2NTI2NjYxNTUyLTMyYmNkZWQ0Y2E5NmQxMTI3ZGM3NDU0NzU4NGE1Y2UwYTVmNTViYjJmNDI0MWZiYTg4ODNmMzgxYTQ3ZmI2OWU) )



## chiTraの引用 / Citing chiTra
chiTraについての論文を発表しています。 / We have published a following paper about chiTra;
- 勝田哲弘, 林政義, 山村崇, Tolmachev Arseny, 高岡一馬, 内田佳孝, 浅原正幸, 単語正規化による表記ゆれに頑健な BERT モデルの構築. 言語処理学会第28回年次大会, 2022.

chiTraを論文や書籍、サービスなどで引用される際には、以下のBibTexをご利用ください。 / When citing chiTra in papers, books, or services, please use the follow BibTex entries;
```
@INPROCEEDINGS{katsuta2022chitra,
    author    = {勝田哲弘, 林政義, 山村崇, Tolmachev Arseny, 高岡一馬, 内田佳孝, 浅原正幸},
    title     = {単語正規化による表記ゆれに頑健な BERT モデルの構築},
    booktitle = "言語処理学会第28回年次大会(NLP2022)",
    year      = "2022",
    pages     = "",
    publisher = "言語処理学会",
}
```

### 実験に使用したモデル / Model used for experiment
「単語正規化による表記ゆれに頑健なBERTモデルの構築」の実験において使用したモデルを以下で公開しています。/  The model used in the experiment of "単語正規化による表記ゆれに頑健なBERTモデルの構築" is published below.

| 　 Normalized          | Text     | Pretrained Model                                                                                                 |
| ---------------------- | -------- | ---------------------------------------------------------------------------------------------------------------- |
| surface                | Wiki-40B | [tar.gz](https://sudachi.s3.ap-northeast-1.amazonaws.com/chitra/nlp2022/Wikipedia_surface.tar.gz)                |
| normalized_and_surface | Wiki-40B | [tar.gz](https://sudachi.s3.ap-northeast-1.amazonaws.com/chitra/nlp2022/Wikipedia_normalized_and_surface.tar.gz) |
| normalized_conjugation | Wiki-40B | [tar.gz](https://sudachi.s3.ap-northeast-1.amazonaws.com/chitra/nlp2022/Wikipedia_normalized_conjugation.tar.gz) |
| normalized             | Wiki-40B | [tar.gz](https://sudachi.s3.ap-northeast-1.amazonaws.com/chitra/nlp2022/Wikipedia_normalized.tar.gz)             |

Enjoy chiTra!
