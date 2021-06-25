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
from glob import glob

from sudachitra.pretokenizer import PartOfSpeechSubstitutionTokenizer


def main():
    args = get_args()

    if args.input_file:
        files = [args.input_file]
    elif args.input_dir:
        files = glob(os.path.join(args.input_dir, '*.txt'))
    else:
        raise ValueError("`input_file` or `input_dir` must be specified.")

    pos_tokenizer = PartOfSpeechSubstitutionTokenizer(
        split_mode=args.split_mode,
        dict_type=args.dict_type,
        word_form_type=args.word_form_type
    )

    pos_tokenizer.train(
        files,
        token_size=args.token_size,
        min_frequency=args.min_frequency,
        limit_character=args.limit_character,
        special_tokens=args.special_tokens
    )

    pos_tokenizer.save_vocab(args.output_file)


def get_args():
    parser = argparse.ArgumentParser(description='Trainer of part-of-speech substitution tokenizer.')

    # Input
    parser.add_argument('-f', '--input_file', default='',
                        help='Input file to train tokenizer.')
    parser.add_argument('-d', '--input_dir', default='',
                        help='Input directory containing files to train tokenizer.')

    # Parameters
    parser.add_argument('--token_size', type=int, default=32000,
                        help='The size of the vocabulary, excluding special tokens and pos tags.')
    parser.add_argument('--min_frequency', type=int, default=1,
                        help='Ignores all words (tokens and characters) with total frequency lower than this.')
    parser.add_argument('--limit_character', type=int, default=1000,
                        help='The maximum different characters to keep in the vocabulary.')
    parser.add_argument('--special_tokens', nargs='*', default=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                        help='A list of special tokens the model should know of.')

    # Tokenization
    parser.add_argument('--dict_type', default='core', choices=['small', 'core', 'full'],
                        help='Sudachi dictionary type to be used for tokenization.')
    parser.add_argument('--split_mode', default='C', choices=['A', 'B', 'C', 'a', 'b', 'c'],
                        help='The mode of splitting.')
    parser.add_argument('--word_form_type', default='surface',
                        choices=['surface', 'dictionary', 'normalized', 'dictionary_and_surface', 'normalized_and_surface'],
                        help='Word form type for each morpheme.')

    # output
    parser.add_argument('-o', '--output_file',
                        help='The output path where the vocabulary will be stored.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
