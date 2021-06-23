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


class SudachipyPreTokenizer(SudachipyWordTokenizer):
    def __init__(
        self,
        split_mode: Optional[str] = "C",
        dict_type: Optional[str] = "core",
        word_form_type: Optional[str] = "surface",
        **kwargs
    ):
        super().__init__(split_mode=split_mode, dict_type=dict_type, **kwargs)
        self.word_form_type = word_form_type
        self.word_formatter = WORD_FORM_TYPES[self.word_form_type] if self.word_form_type != "surface" else None

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.sudachi_split)

    @staticmethod
    def split_normalized_string(normalized_string: NormalizedString, tokens: List[str]) -> List[NormalizedString]:
        token_spans = textspan.get_original_spans(tokens, str(normalized_string).strip())
        return [normalized_string[start:end] for token_span in token_spans for start, end in token_span]

    def sudachi_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        morphs = super().tokenize(str(normalized_string).strip())
        tokens = list(map(lambda m: m.surface(), morphs))
        normalized_strings = self.split_normalized_string(normalized_string, tokens)
        if not (len(morphs) == len(tokens) == len(normalized_strings)):
            raise ValueError(len(morphs), len(tokens), len(normalized_strings), tokens, normalized_strings)

        if self.word_form_type != 'surface':
            _ = [ns.replace(ns.normalized, self.word_formatter(m)) for ns, m in zip(normalized_strings, morphs)]

        return normalized_strings
