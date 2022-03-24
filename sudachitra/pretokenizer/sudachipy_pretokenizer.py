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

from sudachipy import Dictionary, MorphemeList
from tokenizers import NormalizedString
from typing import Callable, List, Optional

from ..word_formatter import word_formatter, WordFormTypes


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
        normalized_strings = []

        for m in morphemes:
            begin_index = m.begin()
            end_index = m.end()
            if begin_index == end_index:  # empty token
                continue

            normalized_string = original[begin_index:end_index]

            if _word_formatter is not None:
                # replace the word form of the `original` string by using `NormalizedString.replace()` with side effect.
                normalized_string.replace(normalized_string.normalized, _word_formatter(m))

            normalized_strings.append(normalized_string)

        return normalized_strings

    return _handler
