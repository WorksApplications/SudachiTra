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


class SentenceFilter(object):

    def is_filtered(self, sentence: str) -> bool:
        """
        Determine if the input sentence should be filtered or not.

        Args:
            sentence (str): A sentence of a document in a training corpus to pretrain model.

        Returns:
            bool: `True` if the sentence should be filtered, otherwise `False`.
        """
        raise NotImplementedError()


SF = TypeVar('SF', bound=SentenceFilter)


class SequenceSentenceFilter(SentenceFilter):
    def __init__(self, sentence_filters: List[SF]):
        """
        Constructs a SequenceSentenceFilter.

        Args:
            sentence_filters (List[SF]): A list of SentenceFilters.
        """
        self._sentence_filters: List[SF] = sentence_filters

    def is_filtered(self, sentence: str) -> bool:
        """
        Applies filters to the sentence in sequence.

        Args:
            sentence (str): A sentence.

        Returns:
            bool:`True` if the sentence should be filtered, otherwise `False`.
        """
        return any([sf.is_filtered(sentence) for sf in self._sentence_filters])


class UrlFilter(SentenceFilter):
    url_pattern = re.compile(r'(https?|sftp?)://[\w/:%#\$&\?\(\)~\.=\+\-]+')

    def is_filtered(self, sentence: str) -> bool:
        """
        Determines if the sentence contains URL.

        Args:
            sentence (str): A sentence.

        Returns:
            bool: `True` if the sentence contains URL, otherwise `False`.
        """
        return bool(self.url_pattern.search(sentence))


class EmailFilter(SentenceFilter):
    mail_pattern = re.compile(r'[\w\d_-]+@[\w\d_-]+\.[\w\d._-]+')

    def is_filtered(self, sentence: str) -> bool:
        """
        Determines if the sentence contains email address.

        Args:
            sentence (str): A sentence.

        Returns:
            bool: `True` if the sentence contains email address, otherwise `False`.
        """
        return bool(self.mail_pattern.search(sentence))


class SequenceLengthFilter(SentenceFilter):

    def __init__(self, min_seq_len: int = 10, max_seq_len: int = 200):
        """
        Constructs a SequenceLengthFilter.

        Args:
            min_seq_len (int): The minimum number of characters a sentence should contain.
            max_seq_len (int): The maximum number of characters a sentence should contain.
        """
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

    def is_filtered(self, sentence: str) -> bool:
        """
        Determines if the number of characters in the sentence is suitable.

        Args:
            sentence (str): A sentence.

        Returns:
            bool: `True` if the sentence length is either too short or too long, otherwise `False`.
        """
        return len(sentence) < self.min_seq_len or self.max_seq_len < len(sentence)
