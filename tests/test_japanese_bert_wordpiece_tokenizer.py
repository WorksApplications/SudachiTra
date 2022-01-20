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

import unittest

from sudachipy import Dictionary, SplitMode
from sudachitra.pretokenizer import JapaneseBertWordPieceTokenizer
from sudachitra.pretokenizer import pretokenizer_handler
from sudachitra.word_formatter import WordFormTypes


class JapaneseBertWordPieceTokenizerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.vocab = {
            "[UNK]": 0,
            "[SEP]": 1,
            "[CLS]": 2,
            "[PAD]": 3,
            "[MASK]": 4,
            "引越": 5,
            "引っ越し": 6,
            "し": 7,
            "て": 8,
            "為る": 9,
            "する": 10,
            "から": 11,
            "すだち": 12,
            "酢橘": 13,
            "-": 14,
            "－": 15,
            "Ｓｕｄａｃｈｉ": 16,
            "Sudachi": 17,
            "sudachi": 18,
            "を": 19,
            "とどけ": 20,
            "とどける": 21,
            "届け": 22,
            "届ける": 23,
            "ます": 24,
            "…": 25,
            ".": 26,
            "。": 27,
            "\n": 28
        }
        self.wp_tokenizer = JapaneseBertWordPieceTokenizer(vocab=self.vocab,
                                                           do_lower_case=False,
                                                           do_nfkc=False,
                                                           do_strip=False)

        self.sudachi_dict = Dictionary(dict='core')
        self.test_sentence = '引越してからすだち－Ｓｕｄａｃｈｉをとどけます。'

    def set_pretokenizer(self, word_form_type: WordFormTypes):
        pretok = self.sudachi_dict.pre_tokenizer(mode=SplitMode.C,
                                                 handler=pretokenizer_handler(self.sudachi_dict,
                                                                              word_form_type=word_form_type))
        self.wp_tokenizer.pre_tokenizer = pretok

    def test_surface(self):
        self.set_pretokenizer(WordFormTypes.SURFACE)

        res = self.wp_tokenizer.encode(self.test_sentence)

        tokens = ['[CLS]', '引越', 'し', 'て', 'から', 'すだち', '－', 'Ｓｕｄａｃｈｉ', 'を', 'とどけ', 'ます', '。', '[SEP]']
        self.assertListEqual(tokens, res.tokens)
        self.assertListEqual(list(map(self.wp_tokenizer.token_to_id, tokens)), res.ids)

    def test_normalized_and_surface(self):
        self.set_pretokenizer(WordFormTypes.NORMALIZED_AND_SURFACE)

        res = self.wp_tokenizer.encode(self.test_sentence)

        tokens = ['[CLS]', '引っ越し', 'し', 'て', 'から', '酢橘', '－', 'Sudachi', 'を', 'とどけ', 'ます', '。', '[SEP]']
        self.assertListEqual(tokens, res.tokens)
        self.assertListEqual(list(map(self.wp_tokenizer.token_to_id, tokens)), res.ids)

    def test_normalized_conjugation(self):
        self.set_pretokenizer(WordFormTypes.NORMALIZED_CONJUGATION)

        res = self.wp_tokenizer.encode(self.test_sentence)
        tokens = ['[CLS]', '引っ越し', 'し', 'て', 'から', '酢橘', '－', 'Sudachi', 'を', '届け', 'ます', '。', '[SEP]']

        self.assertListEqual(tokens, res.tokens)
        self.assertListEqual(list(map(self.wp_tokenizer.token_to_id, tokens)), res.ids)

    def test_normalized_form(self):
        self.set_pretokenizer(WordFormTypes.NORMALIZED)

        res = self.wp_tokenizer.encode(self.test_sentence)

        tokens = ['[CLS]', '引っ越し', '為る', 'て', 'から', '酢橘', '－', 'Sudachi', 'を', '届ける', 'ます', '。', '[SEP]']
        self.assertListEqual(tokens, res.tokens)
        self.assertListEqual(list(map(self.wp_tokenizer.token_to_id, tokens)), res.ids)

    def test_dictionary_form(self):
        self.set_pretokenizer(WordFormTypes.DICTIONARY)

        res = self.wp_tokenizer.encode(self.test_sentence)

        tokens = ['[CLS]', '引越', 'する', 'て', 'から', 'すだち', '－', 'Sudachi', 'を', 'とどける', 'ます', '。', '[SEP]']
        self.assertListEqual(tokens, res.tokens)
        self.assertListEqual(list(map(self.wp_tokenizer.token_to_id, tokens)), res.ids)

    def test_dictionary_and_surface(self):
        self.set_pretokenizer(WordFormTypes.DICTIONARY_AND_SURFACE)

        res = self.wp_tokenizer.encode(self.test_sentence)
        tokens = ['[CLS]', '引越', 'し', 'て', 'から', 'すだち', '－', 'Sudachi', 'を', 'とどけ', 'ます', '。', '[SEP]']
        self.assertListEqual(tokens, res.tokens)
        self.assertListEqual(list(map(self.wp_tokenizer.token_to_id, tokens)), res.ids)

    def test_normalizers(self):
        self.wp_tokenizer = JapaneseBertWordPieceTokenizer(vocab=self.vocab,
                                                           do_lower_case=True,
                                                           do_nfkc=True,
                                                           do_strip=True)
        self.set_pretokenizer(WordFormTypes.SURFACE)

        # lowercase
        res = self.wp_tokenizer.encode('SUDACHI')
        tokens = ['[CLS]', 'sudachi', '[SEP]']
        self.assertListEqual(tokens, res.tokens)
        self.assertListEqual(list(map(self.wp_tokenizer.token_to_id, tokens)), res.ids)

        # strip
        res = self.wp_tokenizer.encode(' sudachi\n')
        tokens = ['[CLS]', 'sudachi', '[SEP]']
        self.assertListEqual(tokens, res.tokens)
        self.assertListEqual(list(map(self.wp_tokenizer.token_to_id, tokens)), res.ids)

        # nfkc
        res = self.wp_tokenizer.encode('…')
        tokens = ['[CLS]', '.', '.', '.', '[SEP]']
        self.assertListEqual(tokens, res.tokens)
        self.assertListEqual(list(map(self.wp_tokenizer.token_to_id, tokens)), res.ids)

    def test_oov(self):
        self.set_pretokenizer(WordFormTypes.SURFACE)

        res = self.wp_tokenizer.encode('OOV')
        tokens = ['[CLS]', '[UNK]', '[SEP]']
        self.assertListEqual(tokens, res.tokens)
        self.assertListEqual(list(map(self.wp_tokenizer.token_to_id, tokens)), res.ids)

