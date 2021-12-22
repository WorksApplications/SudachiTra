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

import argparse
from logzero import logger
from progressbar import progressbar
from typing import List

from corpus_preprocessing.filter.sentence_filter import (
    SF, SentenceFilterName,
    SequenceSentenceFilter,
    EmailFilter, UrlFilter, SequenceLengthFilter
)
from corpus_preprocessing.filter.document_filter import (
    DF, DocumentFilterName,
    SequenceDocumentFilter,
    NGWordsFilter, ShortDocumentFilter, ScriptFilter
)
from corpus_preprocessing.normalizer.sentence_normalizer import (
    SN, SentenceNormalizerName,
    SequenceSentenceNormalizer,
    CitationNormalizer, WhitespaceNormalizer
)

from corpus_preprocessing.normalizer.document_normalizer import (
    DN, DocumentNormalizerName,
    SequenceDocumentNormalizer,
    ConcatShortSentenceNormalizer
)


def load_dataset(input_dataset_path: str) -> List[List[str]]:
    documents = []
    with open(input_dataset_path, 'r', encoding='utf-8') as f:
        document = []
        for sentence in f:
            sentence = sentence.strip()
            if sentence != '':
                document.append(sentence)
            else:
                documents.append(document)
                document = []

    return documents


def load_sentence_filters(sentence_filter_names: List[str], **kwargs) -> List[SF]:
    sentence_filters = []
    for sentence_filter_name in sentence_filter_names:
        if sentence_filter_name == SentenceFilterName.EMAIL:
            sentence_filters.append(EmailFilter())
        elif sentence_filter_name == SentenceFilterName.URL:
            sentence_filters.append(UrlFilter())
        elif sentence_filter_name == SentenceFilterName.SEQUENCE_LENGTH:
            sentence_filters.append(SequenceLengthFilter(min_seq_len=kwargs['min_seq_len'],
                                                         max_seq_len=kwargs['max_seq_len']))
        else:
            raise ValueError('Invalid sentence filter name `{}`: {}'.format(
                sentence_filter_name, ','.join(map(str, SentenceFilterName)))
            )

    return sentence_filters


def load_document_filters(document_filter_names: List[str], **kwargs) -> List[DF]:
    document_filters = []
    for document_filter_name in document_filter_names:
        if document_filter_name == DocumentFilterName.SHORT_DOCUMENT:
            document_filters.append(ShortDocumentFilter(min_sentence_num=kwargs['min_sentence_num']))
        elif document_filter_name == DocumentFilterName.SCRIPT:
            document_filters.append(ScriptFilter())
        elif document_filter_name == DocumentFilterName.NG_WORDS:
            document_filters.append(NGWordsFilter(ng_words_file_path=kwargs['ng_words_file_path']))
        else:
            raise ValueError('Invalid document filter name `{}`: {}'.format(
                document_filter_name, ','.join(map(str, DocumentFilterName)))
            )

    return document_filters


def load_sentence_normalizers(sentence_normalizer_names: List[str], **kwargs) -> List[SN]:
    sentence_normalizers = []
    for sentence_normalizer_name in sentence_normalizer_names:
        if sentence_normalizer_name == SentenceNormalizerName.CITATION:
            sentence_normalizers.append(CitationNormalizer())
        elif sentence_normalizer_name == SentenceNormalizerName.WHITESPACE:
            sentence_normalizers.append(WhitespaceNormalizer())
        else:
            raise ValueError('Invalid sentence normalizer name `{}`: {}'.format(
                sentence_normalizer_name, ','.join(map(str, SentenceNormalizerName)))
            )

    return sentence_normalizers


def load_document_normalizers(document_normalizer_names: List[str], **kwargs) -> List[DN]:
    document_normalizers = []
    for document_normalizer_name in document_normalizer_names:
        if document_normalizer_name == DocumentNormalizerName.CONCAT_SHORT_SENTENCE:
            document_normalizers.append(ConcatShortSentenceNormalizer(concat_char_num=kwargs['concat_char_num']))
        else:
            raise ValueError('Invalid document normalizer name `{}`: {}'.format(
                document_normalizer_name, ','.join(map(str, DocumentNormalizerName)))
            )

    return document_normalizers


def main():
    args = get_args()
    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))

    sequence_sentence_filter = SequenceSentenceFilter(
        load_sentence_filters(args.sentence_filter_names,
                              min_seq_len=args.min_seq_len,
                              max_seq_len=args.max_seq_len)
    )

    sequence_document_filter = SequenceDocumentFilter(
        load_document_filters(args.document_filter_names,
                              min_sentence_num=args.min_sentence_num,
                              ng_words_file_path=args.ng_words_file_path)
    )

    sequence_sentence_normalizer = SequenceSentenceNormalizer(
        load_sentence_normalizers(args.sentence_normalizer_names)
    )

    sequence_document_normalizer = SequenceDocumentNormalizer(
        load_document_normalizers(args.document_normalizer_names,
                                  concat_char_num=args.concat_char_num)
    )

    documents = load_dataset(args.input_dataset_path)
    preprocessed_documents = []
    for document in progressbar(documents):
        # normalize
        document = [sequence_sentence_normalizer.normalize(s) for s in document]
        document = sequence_document_normalizer.normalize(document)

        # filter
        document = [s for s in document if not sequence_sentence_filter.is_filtered(s)]
        if not sequence_document_filter.is_filtered(document):
            preprocessed_documents.append(document)

    logger.info('#Document w/o filtering:\t{}'.format(len(documents)))
    logger.info('#Document w/ filtering:\t{}'.format(len(preprocessed_documents)))

    with open(args.output_dataset_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(['\n'.join(document) for document in preprocessed_documents]))


def get_args():
    parser = argparse.ArgumentParser(description='Cleaning and corpus_preprocessing dataset.')
    parser.add_argument('-i', '--input_dataset_path', required=True,
                        help='Input dataset.')
    parser.add_argument('-o', '--output_dataset_path', required=True,
                        help='Output dataset.')

    # Sentence filters
    parser.add_argument('-sf', '--sentence_filter_names', nargs='*', default=list(),
                        choices=SentenceFilterName, type=SentenceFilterName,
                        help='A list of filter names to remove unnecessary sentences from the training corpus.')
    parser.add_argument('--min_seq_len', type=int, default=10,
                        help='The minimum number of characters a sentence should contain (for SequenceLengthFilter).')
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help='The maximum number of characters a sentence should contain (for SequenceLengthFilter).')

    # Document filters
    parser.add_argument('-df', '--document_filter_names', nargs='*', default=list(),
                        choices=DocumentFilterName, type=DocumentFilterName,
                        help='A list of filter names to remove unnecessary documents from the training corpus.')
    parser.add_argument('--min_sentence_num', type=int, default=5,
                        help='The minimum number of sentences a document should contain (for ShortDocumentFilter).')
    parser.add_argument('--ng_words_file_path',
                        default='./resources/ng_words.txt',
                        help='A file path of NG word list (for NGWordsFilter).')

    # Sentence normalizers
    parser.add_argument('-sn', '--sentence_normalizer_names', nargs='*', default=list(),
                        choices=SentenceNormalizerName, type=SentenceNormalizerName,
                        help='A list of filter names to normalize sentences from the training corpus.')

    # Document normalizers
    parser.add_argument('-dn', '--document_normalizer_names', nargs='*', default=list(),
                        choices=DocumentNormalizerName, type=DocumentNormalizerName,
                        help='A list of filter names to normalize documents from the training corpus.')
    parser.add_argument('--concat_char_num', type=int, default=2,
                        help='The maximum number of characters to be concatenated with the previous sentence '
                             '(for ConcatShortSentenceNormalizer).')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
