# Copyright (c) 2021 Works Applications Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import shutil
import tempfile
import unittest

from transformers.models.bert.tokenization_bert import WordpieceTokenizer

from sudachitra import BertSudachipyTokenizer, SudachipyWordTokenizer
from sudachitra.tokenization_bert_sudachipy import VOCAB_FILES_NAMES, WORD_FORM_TYPES


class BertSudachipyTokenizationTest(unittest.TestCase):

    tokenizer_class = BertSudachipyTokenizer

    def setUp(self):
        super().setUp()
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "こんにちは",
            "こん",
            "にちは",
            "ばんは",
            "##こん",
            "##にちは",
            "##ばんは",
            "世界",
            "##世界",
            "、",
            "##、",
            "。",
            "##。",
        ]

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_pickle_sudachipy_tokenizer(self):
        tokenizer = self.tokenizer_class(
            self.vocab_file,
            do_lower_case=False,
            do_word_tokenize=True,
            subword_tokenizer_type='wordpiece'
        )
        self.assertIsNotNone(tokenizer)

        text = "こんにちは、世界。\nこんばんは、世界。"
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, ["こんにちは", "、", "世界", "。", "こん", "##ばんは", "、", "世界", "。"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [3, 12, 10, 14, 4, 9, 12, 10, 14])

        filename = os.path.join(self.tmpdirname, "tokenizer.bin")
        with open(filename, "wb") as handle:
            pickle.dump(tokenizer, handle)

        with open(filename, "rb") as handle:
            tokenizer_new = pickle.load(handle)

        tokens_loaded = tokenizer_new.tokenize(text)

        self.assertListEqual(tokens, tokens_loaded)

    def test_sudachipy_tokenizer_small(self):
        try:
            tokenizer = SudachipyWordTokenizer(dict_type="small")
        except ModuleNotFoundError:
            return

        self.assertListEqual(
            list(map(WORD_FORM_TYPES['surface'],
                     tokenizer.tokenize("appleはsmall辞書に、apple pieはcore辞書に、apple storeはfull辞書に収録されている。"))),
            ["apple", "は", "small", "辞書", "に", "、", "apple", " ", "pie", "は", "core", "辞書", "に", "、",
             "apple", " ", "store", "は", "full", "辞書", "に", "収録", "さ", "れ", "て", "いる", "。"]
        )

    def test_sudachipy_tokenizer_core(self):
        try:
            tokenizer = SudachipyWordTokenizer(dict_type="core")
        except ModuleNotFoundError:
            return

        self.assertListEqual(
            list(map(WORD_FORM_TYPES['surface'],
                     tokenizer.tokenize("appleはsmall辞書に、apple pieはcore辞書に、apple storeはfull辞書に収録されている。"))),
            ["apple", "は", "small", "辞書", "に", "、", "apple pie", "は", "core", "辞書", "に", "、",
             "apple", " ", "store", "は", "full", "辞書", "に", "収録", "さ", "れ", "て", "いる", "。"]
        )

    def test_sudachipy_tokenizer_full(self):
        try:
            tokenizer = SudachipyWordTokenizer(dict_type="full")
        except ModuleNotFoundError:
            return

        self.assertListEqual(
            list(map(WORD_FORM_TYPES['surface'],
                     tokenizer.tokenize("appleはsmall辞書に、apple pieはcore辞書に、apple storeはfull辞書に収録されている。"))),
            ["apple", "は", "small", "辞書", "に", "、", "apple pie", "は", "core", "辞書", "に", "、",
             "apple store", "は", "full", "辞書", "に", "収録", "さ", "れ", "て", "いる", "。"]
        )

    def test_sudachipy_tokenizer_surface(self):
        tokenizer = self.tokenizer_class(self.vocab_file, do_subword_tokenize=False,
                                         word_form_type='surface')

        self.assertListEqual(
            tokenizer.tokenize("appleの辞書形はAppleで正規形はアップルである。"),
            ["apple", "の", "辞書形", "は", "Apple", "で", "正規", "形", "は", "アップル", "で", "ある", "。"]

        )

    def test_sudachipy_tokenizer_dictionary_form(self):
        tokenizer = self.tokenizer_class(self.vocab_file, do_subword_tokenize=False,
                                         word_form_type='dictionary')

        self.assertListEqual(
            tokenizer.tokenize("appleの辞書形はAppleで正規形はアップルである。"),
            ["Apple", "の", "辞書形", "は", "Apple", "で", "正規", "形", "は", "アップル", "だ", "ある", "。"]
        )

    def test_sudachipy_tokenizer_normalized_form(self):
        tokenizer = self.tokenizer_class(self.vocab_file, do_subword_tokenize=False,
                                         word_form_type='normalized')

        self.assertListEqual(
            tokenizer.tokenize("appleの辞書形はAppleで正規形はアップルである。"),
            ["アップル", "の", "辞書形", "は", "アップル", "で", "正規", "形", "は", "アップル", "だ", "有る", "。"]
        )

    def test_sudachipy_tokenizer_dictionary_form_and_surface(self):
        tokenizer = self.tokenizer_class(self.vocab_file, do_subword_tokenize=False,
                                         word_form_type='dictionary_and_surface')

        self.assertListEqual(
            tokenizer.tokenize("appleの辞書形はAppleで正規形はアップルである。"),
            ["Apple", "の", "辞書形", "は", "Apple", "で", "正規", "形", "は", "アップル", "で", "ある", "。"]
        )

    def test_sudachipy_tokenizer_normalized_form_and_surface(self):
        tokenizer = self.tokenizer_class(self.vocab_file, do_subword_tokenize=False,
                                         word_form_type='normalized_and_surface')

        self.assertListEqual(
            tokenizer.tokenize("appleの辞書形はAppleで正規形はアップルである。"),
            ["アップル", "の", "辞書形", "は", "アップル", "で", "正規", "形", "は", "アップル", "で", "ある", "。"]
        )

    def test_sudachipy_tokenizer_unit_a(self):
        try:
            tokenizer = SudachipyWordTokenizer(split_mode="A")
        except ModuleNotFoundError:
            return

        self.assertListEqual(
            list(map(WORD_FORM_TYPES['surface'], tokenizer.tokenize("徳島阿波おどり空港"))),
            ["徳島", "阿波", "おどり", "空港"]
        )

    def test_sudachipy_tokenizer_unit_b(self):
        try:
            tokenizer = SudachipyWordTokenizer(split_mode="B")
        except ModuleNotFoundError:
            return

        self.assertListEqual(
            list(map(WORD_FORM_TYPES['surface'], tokenizer.tokenize("徳島阿波おどり空港"))),
            ["徳島", "阿波おどり", "空港"]
        )

    def test_sudachipy_tokenizer_unit_c(self):
        try:
            tokenizer = SudachipyWordTokenizer(split_mode="C")
        except ModuleNotFoundError:
            return

        self.assertListEqual(
            list(map(WORD_FORM_TYPES['surface'], tokenizer.tokenize("徳島阿波おどり空港"))),
            ["徳島阿波おどり空港"]
        )

    def test_wordpiece_tokenizer(self):
        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "こんにちは", "こん", "にちは" "ばんは", "##こん", "##にちは", "##ばんは"]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = WordpieceTokenizer(vocab=vocab, unk_token="[UNK]")

        self.assertListEqual(tokenizer.tokenize(""), [])

        self.assertListEqual(tokenizer.tokenize("こんにちは"), ["こんにちは"])

        self.assertListEqual(tokenizer.tokenize("こんばんは"), ["こん", "##ばんは"])

        self.assertListEqual(tokenizer.tokenize("こんばんは こんばんにちは こんにちは"), ["こん", "##ばんは", "[UNK]", "こんにちは"])

    def test_sequence_builders(self):
        pass


class BertSudachipyCharacterTokenizationTest(unittest.TestCase):

    tokenizer_class = BertSudachipyTokenizer

    def setUp(self):
        super().setUp()
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "こ", "ん", "に", "ち", "は", "ば", "世", "界", "、", "。"]

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class(self.vocab_file, subword_tokenizer_type="character")

        tokens = tokenizer.tokenize("こんにちは、世界。こんばんは、世界。")
        self.assertListEqual(
            tokens, ["こ", "ん", "に", "ち", "は", "、", "世", "界", "。", "こ", "ん", "ば", "ん", "は", "、", "世", "界", "。"]
        )
        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens), [3, 4, 5, 6, 7, 11, 9, 10, 12, 3, 4, 8, 4, 7, 11, 9, 10, 12]
        )

    def test_sequence_builders(self):
        pass
