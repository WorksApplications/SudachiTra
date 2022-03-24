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
import os
import tensorflow_datasets as tfds
from bunkai import Bunkai
from progressbar import progressbar as tqdm
from tensorflow_datasets.core import DatasetInfo
from typing import List, Tuple


TARGET_DATASETS = ['train', 'validation', 'test']

START_ARTICLE_DELIMITER = '_START_ARTICLE_'
START_PARAGRAPH_DELIMITER = '_START_PARAGRAPH_'
NEW_LINE_DELIMITER = '_NEWLINE_'


def split_article(article_text: str) -> List[List[str]]:
    """
    Splits an article into paragraphs.

    Args: article_text (str): An article in wikipedia.

    Returns:
        List[List[str]]: List of paragraphs containing sentences
    """
    paragraphs = []
    lines = article_text.split('\n')
    for i in range(2, len(lines), 2):
        if lines[i-1] == START_PARAGRAPH_DELIMITER:
            paragraphs.append(lines[i].split(NEW_LINE_DELIMITER))

    return paragraphs


def download_wiki40b_corpus(target: str) -> Tuple[DatasetInfo, List[str]]:
    """
    Downloads the target corpus and disambiguates sentence boundaries for sentences in the corpus.

    Args:
        target (str): Target dataset name.

    Returns:
        (tuple): tuple containing:
            ds_info (DatasetInfo): Dataset information for target corpus.
            all_sentences (List[str]): Sentences in the target corpus.
    """

    ds, ds_info = tfds.load(name='wiki40b/ja', split=TARGET_DATASETS, with_info=True)

    bunkai = Bunkai()

    all_sentences = []
    for line in tqdm(tfds.as_dataframe(ds[TARGET_DATASETS.index(target)], ds_info).itertuples()):
        paragraphs = split_article(line.text.decode('utf-8'))
        all_sentences.append(START_ARTICLE_DELIMITER)
        for paragraph in paragraphs:
            all_sentences.append(START_PARAGRAPH_DELIMITER)
            for sentences in paragraph:
                for sentence in bunkai(sentences):
                    if sentence:
                        all_sentences.append(sentence)

    return ds_info, all_sentences


def main():
    args = get_args()

    dataset_info, corpus_sentences = download_wiki40b_corpus(args.target)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'dataset_info_{}.json'.format(args.target)), 'w') as f:
        f.write(dataset_info.as_json)

    with open(os.path.join(args.output_dir, 'ja_wiki40b_{}.txt'.format(args.target)), 'w') as f:
        for sentence in corpus_sentences:
            f.write(sentence + '\n')


def get_args():
    parser = argparse.ArgumentParser(description='Download and parse target dataset.')
    parser.add_argument('-t', '--target', choices=['train', 'validation', 'test'], help='Target dataset.')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='The path to the target directory in which to save a corpus file and a config file.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
