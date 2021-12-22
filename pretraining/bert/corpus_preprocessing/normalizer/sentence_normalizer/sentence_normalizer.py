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
from typing import List, TypeVar


class SentenceNormalizer(object):

    def normalize(self, sentence: str) -> str:
        """
        Normalizes the sentence.

        Args:
            sentence (str): A sentence.

        Returns:
            str: A normalized sentence.
        """
        raise NotImplementedError()


SN = TypeVar('SN', bound=SentenceNormalizer)


class SequenceSentenceNormalizer(SentenceNormalizer):
    def __init__(self, sentence_normalizers: List[SN]):
        """
        Constructs a SequenceSentenceNormalizer.

        Args:
            sentence_normalizers (List[SN]): A list of SentenceNormalizers.
        """
        self._sentence_normalizers: List[SN] = sentence_normalizers

    def normalize(self, sentence: str) -> str:
        """
        Applies normalizers to the sentence in sequence.

        Args:
            sentence (str): A sentence.

        Returns:
            str: A normalized sentence.
        """
        for sentence_normalizer in self._sentence_normalizers:
            sentence = sentence_normalizer.normalize(sentence)

        return sentence


class WhitespaceNormalizer(SentenceNormalizer):
    continuous_whitespace_pattern = re.compile(r'\s+')

    def normalize(self, sentence: str) -> str:
        """
        Removes invisible characters and replaces consecutive whitespace with a single whitespace.

        Args:
            sentence (str): A sentence.

        Returns:
            str: A sentence with consecutive whitespace and invisible characters removed.
        """
        sentence = "".join(c for c in sentence if c.isprintable())
        sentence = self.continuous_whitespace_pattern.sub(' ', sentence)

        return sentence


class CitationNormalizer(SentenceNormalizer):
    citation_pattern = re.compile(r'\[\d+?\]|\[要.+?\]|\{\{+[^{}]+?\}\}+|\[(要出典|リンク切れ|.+?\?)\]')

    def normalize(self, sentence: str) -> str:
        """
        Removes citation markers.

        Args:
            sentence (str): A sentence.

        Returns:
            str: A sentence with citation markers removed.
        """
        return self.citation_pattern.sub('', sentence)
