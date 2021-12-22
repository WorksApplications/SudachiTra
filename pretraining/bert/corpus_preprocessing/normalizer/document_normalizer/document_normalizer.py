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

from typing import List, TypeVar


class DocumentNormalizer(object):

    def normalize(self, document: List[str]) -> List[str]:
        """
        Normalizes the document.

        Args:
            document (List[str]): A list of sentences.

        Returns:
            List[str]: A normalized document.
        """
        raise NotImplementedError()


DN = TypeVar('DN', bound=DocumentNormalizer)


class SequenceDocumentNormalizer(DocumentNormalizer):
    def __init__(self, document_normalizers: List[DN]):
        """
        Constructs a SequenceDocumentNormalizer.

        Args:
            document_normalizers (List[DN]): A list of DocumentNormalizers.
        """
        self._document_normalizers: List[DN] = document_normalizers

    def normalize(self, document: List[str]) -> List[str]:
        """
        Applies normalizers to the document in sequence.

        Args:
            document (List[str]): A list of sentences.

        Returns:
            List[str]: A normalized document.
        """
        for document_normalizer in self._document_normalizers:
            document = document_normalizer.normalize(document)

        return document


class ConcatShortSentenceNormalizer(DocumentNormalizer):

    def __init__(self, concat_char_num: int = 2):
        """
        Constructs a ConcatShortSentenceNormalizer.

        Args:
            concat_char_num (int): The maximum number of characters to be concatenated with the previous sentence.
        """
        self.concat_char_num = concat_char_num

    def normalize(self, document: List[str]) -> List[str]:
        """
        Joins a short sentence that are only a few characters to the previous sentence.

        Args:
            document (List[str]): A list of sentences.

        Returns:
            List[str]: A document with short sentences concatenated.
        """

        if len(document) == 1:
            return document
        else:
            concat_ids = []
            for i, sentence in enumerate(document):
                if 0 < i and len(sentence) <= self.concat_char_num:
                    concat_ids.append(i)

            for concat_id in concat_ids[::-1]:
                document[concat_id - 1] += document.pop(concat_id)
            return document
