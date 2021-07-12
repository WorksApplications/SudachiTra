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

import textspan
from tokenizers import NormalizedString, PreTokenizedString
from typing import List, Optional

from .. import SudachipyWordTokenizer
from ..tokenization_bert_sudachipy import WORD_FORM_TYPES


class CustomPreTokenizer:
    def custom_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """
        Tokenizes the input string and returns list of tokens.

        Please override this function with your custom tokenizer.
        Example. https://github.com/huggingface/tokenizers/blob/b24a2fc/bindings/python/examples/custom_components.py

        Args:
            i (int): Index.
            normalized_string (NormalizedString): Input string.

        Returns:
            List[NormalizedString]: List of normalized_strings.
        """
        raise NotImplementedError()

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)

    @staticmethod
    def split_normalized_string(normalized_string: NormalizedString, tokens: List[str]) -> List[NormalizedString]:
        """
        Splits normalized_string by tokens.

        Args:
            normalized_string (NormalizedString): Input string.
            tokens (List[str]): List of surface words in a sentence.

        Returns:
            List[NormalizedString]: List of normalized_strings.
        """
        token_spans = textspan.get_original_spans(tokens, str(normalized_string).strip())
        return [normalized_string[start:end] for token_span in token_spans for start, end in token_span]


class SudachipyPreTokenizer(SudachipyWordTokenizer, CustomPreTokenizer):
    def __init__(
        self,
        split_mode: Optional[str] = "C",
        dict_type: Optional[str] = "core",
        word_form_type: Optional[str] = "surface",
        **kwargs
    ):
        """
        Constructs a SudachipyPreTokenizer.

        Args:
            split_mode (:obj:`str`, `optional`, defaults to :obj:`"C"`):
                The mode of splitting.
                "A", "B", or "C" can be specified.
            dict_type (:obj:`str`, `optional`, defaults to :obj:`"core"`):
                Sudachi dictionary type to be used for tokenization.
                "small", "core", or "full" can be specified.
            word_form_type (:obj:`str`, `optional`, defaults to :obj:`"surface"`):
                Word form type for each morpheme.
                "surface", "dictionary", "normalized", "dictionary_and_surface", or "normalized_and_surface" can be specified.
            **kwargs:
                Sudachi dictionary parameters.
        """
        SudachipyWordTokenizer.__init__(self, split_mode=split_mode, dict_type=dict_type, **kwargs)
        self.word_form_type = word_form_type
        self.word_formatter = WORD_FORM_TYPES[self.word_form_type] if self.word_form_type != "surface" else None

    def custom_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """
        Tokenizes with SudachiPy and returns its tokens.

        Args:
            i (int): Index.
            normalized_string (NormalizedString): Input string.

        Returns:
            List[NormalizedString]: List of normalized_strings.
        """
        morphs = super().tokenize(str(normalized_string).strip())
        tokens = list(map(lambda m: m.surface(), morphs))
        normalized_strings = self.split_normalized_string(normalized_string, tokens)
        if not (len(morphs) == len(tokens) == len(normalized_strings)):
            raise ValueError(len(morphs), len(tokens), len(normalized_strings), tokens, normalized_strings)

        if self.word_form_type != 'surface':
            _ = [ns.replace(ns.normalized, self.word_formatter(m)) for ns, m in zip(normalized_strings, morphs)]

        return normalized_strings
