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
from sudachipy import Dictionary, MorphemeList
from tokenizers import NormalizedString
from typing import Callable, List, Optional

from ..word_formatter import word_formatter, WordFormTypes


def split_normalized_string(normalized_string: NormalizedString, tokens: List[str]) -> List[NormalizedString]:
    """
    Splits normalized_string by tokens.

    Args:
        normalized_string (NormalizedString): Input string.
        tokens (List[str]): List of surface words in a sentence.

    Returns:
        List[NormalizedString]: List of normalized_strings.
    """
    token_spans = textspan.get_original_spans(tokens, str(normalized_string))
    return [normalized_string[start:end] for token_span in token_spans for start, end in token_span]


def pretokenizer_handler(sudachi_dict: Dictionary, word_form_type: Optional[str] = 'surface')\
        -> Callable[[int, NormalizedString, MorphemeList], List[NormalizedString]]:
    """
    A handler for Dictionary.pre_tokenizer that transform MorphemeList into list to tokens.

    Returns a handler to convert a morpheme to the specified word form.

    Args:
        sudachi_dict (Dictionary):
            Sudachi dictionary.
        word_form_type (:obj:`str`, `optional`, defaults to :obj:`"surface"`):
            Word form type for each morpheme.
            The values defined in WordFormTypes can be specified.

    Returns:
        Callable[[int, NormalizedString, MorphemeList], List[NormalizedString]]:
            A handler for Dictionary.pre_tokenizer that transform MorphemeList into list to tokens.
            https://worksapplications.github.io/sudachi.rs/python/api/sudachipy.html#sudachipy.Dictionary.pre_tokenizer
    """
    _word_formatter = word_formatter(word_form_type, sudachi_dict) if word_form_type != WordFormTypes.SURFACE else None

    def _handler(index: int, original: NormalizedString, morphemes: MorphemeList) -> List[NormalizedString]:
        tokens = [m.surface() for m in morphemes if m.surface() != '']
        normalized_strings = split_normalized_string(original, tokens)
        if len(tokens) != len(normalized_strings):
            raise ValueError(len(morphemes), len(tokens), len(normalized_strings), tokens, morphemes, normalized_strings)
        if word_form_type != WordFormTypes.SURFACE:
            _ = [ns.replace(ns.normalized, _word_formatter(m)) for ns, m in zip(normalized_strings, morphemes)]

        return normalized_strings

    return _handler
