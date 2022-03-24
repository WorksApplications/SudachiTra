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
from re import Match
from sudachipy import Dictionary, SplitMode
from typing import List, TypeVar


class DocumentFilter(object):

    def is_filtered(self, document: List[str]) -> bool:
        """
        Determine if the input document should be filtered or not.

        Args:
            document (List[str]): Sentences in a training corpus to pretrain model.

        Returns:
            bool: `True` if the document should be filtered, otherwise `False`.
        """
        raise NotImplementedError()


DF = TypeVar('DF', bound=DocumentFilter)


class SequenceDocumentFilter(DocumentFilter):
    def __init__(self, document_filters: List[DF]):
        """
        Constructs a `SequenceDocumentFilter`.

        Args:
            document_filters (List[DF]): A list of DocumentFilters.
        """
        self._document_filters: List[DF] = document_filters

    def is_filtered(self, document: List[str]) -> bool:
        """
        Applies filters to the document in sequence.

        Args:
            document (List[str]): A list of sentences.

        Returns:
            bool: `True` if the document should be filtered, otherwise `False`.
        """
        return any([df.is_filtered(document) for df in self._document_filters])


class ScriptFilter(DocumentFilter):
    curly_brackets_ptn = re.compile(r'[\{|\}]')

    def is_filtered(self, document: List[str]) -> bool:
        """
        Determines if the document contains the curly bracket ```{``` to if it contains code.

        Args:
            document (List[str]): A list of sentences.

        Returns:
            bool: `True` if the document contains the curly bracket, otherwise `False`.
        """
        return any([self.curly_brackets_ptn.search(sentence) for sentence in document])


class ShortDocumentFilter(DocumentFilter):

    def __init__(self, min_sentence_num: int = 5):
        """
        Constructs a ShortDocumentFilter.

        Args:
            min_sentence_num (int): The minimum number of sentences a document should contain.
        """
        self.min_sentence_num = min_sentence_num

    def is_filtered(self, document: List[str]) -> bool:
        """
        Determines if the number of sentences in the document is suitable.

        Args:
            document (List[str]): A list of sentences.

        Returns:
            bool: `True` if the document is short, otherwise `False`.
        """
        return len(document) < self.min_sentence_num


class NGWordsFilter(DocumentFilter):
    DICT_TYPE = 'core'
    SPLIT_MODE = SplitMode.A

    def __init__(self, ng_words_file_path: str):
        """
        Constructs a NGWordsFilter.

        Args:
            ng_words_file_path (str): A file path of NG word list.
        """
        self.ng_words_file_path = ng_words_file_path
        with open(self.ng_words_file_path, 'r', encoding='utf-8') as f:
            ng_words = [line.rstrip() for line in f if line.strip() != '']
        self.ng_words_ptn = re.compile(r'({})'.format('|'.join(ng_words)))

        self.sudachi = Dictionary(dict=self.DICT_TYPE).create(self.SPLIT_MODE)

    def is_matched_by_morpheme(self, match: Match, sentence: str) -> bool:
        """
        Determines if a substring in the sentence matches at the morphological level.

        Args:
            match (Match): A Match object.
            sentence (str): A sentence.

        Returns:
            bool: `True` if a substring is included in the sentence as a word, otherwise `False`.
        """
        matched_begin_id, matched_end_id = match.span()

        morph_begin_ids = set()
        morph_end_ids = set()
        for m in self.sudachi.tokenize(sentence):
            morph_begin_id, morph_end_id = m.begin(), m.end()
            if morph_begin_id <= matched_begin_id:
                morph_begin_ids.add(morph_begin_id)
                morph_end_ids.add(morph_end_id)
            else:
                break

        return matched_begin_id in morph_begin_ids and matched_end_id in morph_end_ids

    def contain_ng_words(self, sentence: str) -> bool:
        """
        Determines if the sentence contains NG words.

        Args:
            sentence (str): A sentence.

        Returns:
            bool: `True` if the sentence contains NG words, otherwise `False`.
        """
        matches = [match for match in self.ng_words_ptn.finditer(sentence)]
        if matches:
            return any([self.is_matched_by_morpheme(match, sentence) for match in matches])
        else:
            return False

    def is_filtered(self, document: List[str]) -> bool:
        """
        Determines if the document contains even a single NG word.

        Args:
            document (List[str]): A list of sentences.

        Returns:
            bool: `True` if the document contains NG words, otherwise `False`.
        """
        return any([self.contain_ng_words(sentence) for sentence in document])
