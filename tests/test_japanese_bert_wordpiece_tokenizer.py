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

import re
import unittest
from typing import List

from sudachipy import Dictionary, SplitMode
from sudachitra.pretokenizer import JapaneseBertWordPieceTokenizer
from sudachitra.pretokenizer import pretokenizer_handler
from sudachitra.word_formatter import WordFormTypes
from tokenizers import Encoding


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
            "とど": 20,
            "届": 21,
            "##け": 22,
            "##る": 23,
            # "届け": 22,
            # "届ける": 23,
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
        self.wordpieces_prefix = '##'
        self.unk_token = '[UNK]'
        self.prefix_pattern = re.compile(f'^{self.wordpieces_prefix}')
        self.delete_prefix = lambda x: self.prefix_pattern.sub('', x)

        self.sudachi_dict = Dictionary(dict='core')
        self.test_sentence = '引越してからすだちＳｕｄａｃｈｉをとどけます。'
        self.sudachi = self.sudachi_dict.create(mode=SplitMode.C)

    def set_pretokenizer(self, word_form_type: WordFormTypes):
        pretok = self.sudachi_dict.pre_tokenizer(mode=SplitMode.C,
                                                 handler=pretokenizer_handler(self.sudachi_dict,
                                                                              word_form_type=word_form_type))
        self.wp_tokenizer.pre_tokenizer = pretok

    def validate_encoding(self, tokens: List[str], encoding: Encoding):
        """
        Validates properties of `Encoding`.
        (https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/__init__.pyi#L69)

        Args:
            tokens (List[str]): Expected tokens.
            encoding (Encoding): Encoded tokens.
        """
        self.assertListEqual(tokens, encoding.tokens)
        self.assertListEqual(list(map(self.wp_tokenizer.token_to_id, tokens)), encoding.ids)
        self.assertListEqual([None, *[0 for _ in range(len(tokens) - 2)], None], encoding.sequence_ids)
        self.assertListEqual([1, *[0 for _ in range(len(tokens) - 2)], 1], encoding.special_tokens_mask)
        self.assertListEqual([0 for _ in range(len(tokens))], encoding.type_ids)
        self.assertListEqual([1 for _ in range(len(tokens))], encoding.attention_mask)
        # ToDo: add test for encoding.offsets and encoding.word_ids (https://github.com/WorksApplications/SudachiTra/issues/42)

    def test_surface(self):
        word_form_type = WordFormTypes.SURFACE
        tokens = ['[CLS]', '引越', 'し', 'て', 'から', 'すだち', 'Ｓｕｄａｃｈｉ', 'を', 'とど', '##け', 'ます', '。', '[SEP]']

        self.set_pretokenizer(word_form_type)
        encoding = self.wp_tokenizer.encode(self.test_sentence)
        self.validate_encoding(tokens, encoding)

    def test_normalized_and_surface(self):
        word_form_type = WordFormTypes.NORMALIZED_AND_SURFACE
        tokens = ['[CLS]', '引っ越し', 'し', 'て', 'から', '酢橘', 'Sudachi', 'を', 'とど', '##け', 'ます', '。', '[SEP]']

        self.set_pretokenizer(word_form_type)
        encoding = self.wp_tokenizer.encode(self.test_sentence)
        self.validate_encoding(tokens, encoding)

    def test_normalized_nouns(self):
        word_form_type = WordFormTypes.NORMALIZED_NOUNS
        tokens = ['[CLS]', '引っ越し', 'し', 'て', 'から', '酢橘', 'Sudachi', 'を', 'とど', '##け', 'ます', '。', '[SEP]']

        self.set_pretokenizer(word_form_type)
        encoding = self.wp_tokenizer.encode(self.test_sentence)
        self.validate_encoding(tokens, encoding)


    def test_normalized_conjugation(self):
        word_form_type = WordFormTypes.NORMALIZED_CONJUGATION
        tokens = ['[CLS]', '引っ越し', 'し', 'て', 'から', '酢橘', 'Sudachi', 'を', '届', '##け', 'ます', '。', '[SEP]']

        self.set_pretokenizer(word_form_type)
        encoding = self.wp_tokenizer.encode(self.test_sentence)
        self.validate_encoding(tokens, encoding)

    def test_normalized_form(self):
        word_form_type = WordFormTypes.NORMALIZED
        tokens = ['[CLS]', '引っ越し', '為る', 'て', 'から', '酢橘', 'Sudachi', 'を', '届', '##け', '##る', 'ます', '。', '[SEP]']

        self.set_pretokenizer(word_form_type)
        encoding = self.wp_tokenizer.encode(self.test_sentence)
        self.validate_encoding(tokens, encoding)

    def test_dictionary_form(self):
        word_form_type = WordFormTypes.DICTIONARY
        tokens = ['[CLS]', '引越', 'する', 'て', 'から', 'すだち', 'Sudachi', 'を', 'とど', '##け', '##る', 'ます', '。', '[SEP]']

        self.set_pretokenizer(word_form_type)
        encoding = self.wp_tokenizer.encode(self.test_sentence)
        self.validate_encoding(tokens, encoding)

    def test_dictionary_and_surface(self):
        word_form_type = WordFormTypes.DICTIONARY_AND_SURFACE
        tokens = ['[CLS]', '引越', 'し', 'て', 'から', 'すだち', 'Sudachi', 'を', 'とど', '##け', 'ます', '。', '[SEP]']

        self.set_pretokenizer(word_form_type)
        encoding = self.wp_tokenizer.encode(self.test_sentence)
        self.validate_encoding(tokens, encoding)

    def test_normalizers(self):
        self.wp_tokenizer = JapaneseBertWordPieceTokenizer(vocab=self.vocab,
                                                           do_lower_case=True,
                                                           do_nfkc=True,
                                                           do_strip=True)
        word_form_type = WordFormTypes.SURFACE
        self.set_pretokenizer(word_form_type)

        # lowercase
        sentence = 'SUDACHI'
        encoding = self.wp_tokenizer.encode(sentence)
        tokens = ['[CLS]', 'sudachi', '[SEP]']
        self.validate_encoding(tokens, encoding)

        # # strip
        sentence = ' sudachi\n'
        encoding = self.wp_tokenizer.encode(sentence)
        tokens = ['[CLS]', 'sudachi', '[SEP]']
        self.validate_encoding(tokens, encoding)
        self.validate_encoding(tokens, encoding)

        # nfkc
        sentence = '…'
        encoding = self.wp_tokenizer.encode(sentence)
        tokens = ['[CLS]', '.', '.', '.', '[SEP]']
        self.validate_encoding(tokens, encoding)
        self.validate_encoding(tokens, encoding)

    def test_oov(self):
        word_form_type = WordFormTypes.SURFACE
        self.set_pretokenizer(word_form_type)

        sentence = 'OOV酢橘OOV酢橘OOV'
        encoding = self.wp_tokenizer.encode(sentence)
        tokens = ['[CLS]', '[UNK]', '酢橘', '[UNK]', '酢橘', '[UNK]', '[SEP]']
        self.validate_encoding(tokens, encoding)
