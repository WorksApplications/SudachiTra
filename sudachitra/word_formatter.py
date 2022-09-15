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
from enum import Enum
from typing import Callable

from sudachipy import Dictionary, Morpheme


HALF_ASCII_TRANSLATE_TABLE = str.maketrans({chr(0xFF01 + _): chr(0x21 + _) for _ in range(94)})

CONJUGATIVE_POS = {'動詞', '形容詞', '助動詞'}


class WordFormTypes(str, Enum):
    SURFACE = 'surface'
    DICTIONARY = 'dictionary'
    NORMALIZED = 'normalized'
    DICTIONARY_AND_SURFACE = 'dictionary_and_surface'
    NORMALIZED_AND_SURFACE = 'normalized_and_surface'
    SURFACE_HALF_ASCII = 'surface_half_ascii'
    DICTIONARY_HALF_ASCII = 'dictionary_half_ascii'
    DICTIONARY_AND_SURFACE_HALF_ASCII = 'dictionary_and_surface_half_ascii'
    NORMALIZED_CONJUGATION = 'normalized_conjugation'
    NORMALIZED_NOUNS = "normalized_nouns"

    def __str__(self):
        return self.value


def word_formatter(word_form_type, sudachi_dict: Dictionary) -> Callable[[Morpheme], str]:
    """
    Returns the function that converts a morpheme to the specified word form.

    Args:
        word_form_type (str): Word form.
        sudachi_dict (Dictionary): Sudachi dictionary.

    Returns:
        Callable[[Morpheme], str]: The function that converts a morpheme to the specified word form.
    """

    if word_form_type not in list(WordFormTypes):
        raise ValueError('Invalid word_form_type error `{}`: {}'.format(word_form_type,
                                                                        list(map(str, WordFormTypes))))

    if word_form_type == WordFormTypes.NORMALIZED_CONJUGATION:
        from sudachitra.conjugation_preserving_normalizer import ConjugationPreservingNormalizer
        conjugation_preserving_normalizer = ConjugationPreservingNormalizer(
            os.path.join(os.path.dirname(__file__), "resources/inflection_table.json"),
            os.path.join(os.path.dirname(__file__), "resources/conjugation_type_table.json"),
            sudachi_dict)

    conjugation_matcher = sudachi_dict.pos_matcher(lambda p: p[0] in CONJUGATIVE_POS)
    nouns_matcher = sudachi_dict.pos_matcher(lambda x: x[5] == "*")

    word_formatters = {
        WordFormTypes.SURFACE: (
            lambda m: m.surface()
        ),
        WordFormTypes.DICTIONARY: (
            lambda m: m.dictionary_form()
        ),
        WordFormTypes.NORMALIZED: (
            lambda m: m.normalized_form()
        ),
        WordFormTypes.DICTIONARY_AND_SURFACE: (
            lambda m: m.surface() if conjugation_matcher(m) else m.dictionary_form()
        ),
        WordFormTypes.NORMALIZED_AND_SURFACE: (
            lambda m: m.surface() if conjugation_matcher(m) else m.normalized_form()
        ),
        WordFormTypes.NORMALIZED_NOUNS: (
            lambda m: m.normalized_form() if nouns_matcher(m) else m.surface()
        ),
        WordFormTypes.SURFACE_HALF_ASCII: (
            lambda m: m.surface().translate(HALF_ASCII_TRANSLATE_TABLE)
        ),
        WordFormTypes.DICTIONARY_HALF_ASCII: (
            lambda m: m.dictionary_form().translate(HALF_ASCII_TRANSLATE_TABLE)
        ),
        WordFormTypes.DICTIONARY_AND_SURFACE_HALF_ASCII: (
            lambda m: m.surface().translate(HALF_ASCII_TRANSLATE_TABLE) if conjugation_matcher(m)
            else m.dictionary_form().translate(HALF_ASCII_TRANSLATE_TABLE)
        ),
        WordFormTypes.NORMALIZED_CONJUGATION: (
            lambda m: conjugation_preserving_normalizer.normalized(m)
        )
    }

    return word_formatters[word_form_type]
