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
import tensorflow_datasets as tfds
from bunkai import Bunkai
from typing import List


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


def main():
    args = get_args()

    ds, ds_info = tfds.load(name='wiki40b/ja', split=['train', 'validation', 'test'], with_info=True)

    bunkai = Bunkai()

    for line in tfds.as_dataframe(ds[TARGET_DATASETS.index(args.target)], ds_info).itertuples():
        paragraphs = split_article(line.text.decode('utf-8'))
        print(START_ARTICLE_DELIMITER)
        for paragraph in paragraphs:
            print(START_PARAGRAPH_DELIMITER)
            for sentences in paragraph:
                for sentence in bunkai(sentences):
                    if sentence:
                        print(sentence)


def get_args():
    parser = argparse.ArgumentParser(description='Download and parse target dataset.')
    parser.add_argument('-t', '--target', choices=['train', 'validation', 'test'], help='Target dataset.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
