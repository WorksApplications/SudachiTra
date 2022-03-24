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

from sudachitra import get_split_mode
from sudachitra.pretokenizer import JapaneseBertWordPieceTokenizer
from sudachitra.word_formatter import WordFormTypes
from sudachitra.pretokenizer import pretokenizer_handler
from sudachipy import Dictionary


def main():
    args = get_args()

    if args.input_file:
        files = [args.input_file]
    elif args.input_dir:
        files = glob(os.path.join(args.input_dir, '*.txt'))
    else:
        raise ValueError("`input_file` or `input_dir` must be specified.")

    settings = dict(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        limit_alphabet=args.limit_alphabet
    )

    wp_tokenizer = JapaneseBertWordPieceTokenizer(do_strip=args.do_strip,
                                                  do_lower_case=args.do_lower_case,
                                                  do_nfkc=args.do_nfkc)

    sudachi_dict = Dictionary(dict=args.dict_type)
    sudachi_pre_tokenizer = sudachi_dict.pre_tokenizer(
        mode=get_split_mode(args.split_mode),
        handler=pretokenizer_handler(sudachi_dict, word_form_type=args.word_form_type)
    )
    wp_tokenizer.pre_tokenizer = sudachi_pre_tokenizer

    if args.disable_parallelism:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    wp_tokenizer.train(files, **settings)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    os.makedirs(args.output_dir, exist_ok=True)
    wp_tokenizer.save(os.path.join(args.output_dir, args.config_name))
    wp_tokenizer.save_vocab(args.output_dir, args.vocab_prefix)


def get_args():
    parser = argparse.ArgumentParser(description='Trainer of wordpiece tokenizer.')

    # input
    parser.add_argument('-f', '--input_file', default='',
                        help='Input file to train tokenizer.')
    parser.add_argument('-d', '--input_dir', default='',
                        help='Input directory containing files to train tokenizer.')

    # Normalizers
    parser.add_argument('--do_strip', action='store_true', default=False,
                        help='Removes all whitespace characters on both sides of the input.')
    parser.add_argument('--do_lower_case', action='store_true', default=False,
                        help='Replaces all uppercase to lowercase.')
    parser.add_argument('--do_nfkc', action='store_true', default=False,
                        help='NFKC unicode normalization.')

    # Parameters
    parser.add_argument('--vocab_size', type=int, default=32000,
                        help='The size of the final vocabulary, including all tokens and alphabet.')
    parser.add_argument('--min_frequency', type=int, default=1,
                        help='The minimum frequency a pair should have in order to be merged.')
    parser.add_argument('--limit_alphabet', type=int, default=5000,
                        help='The maximum different characters to keep in the alphabet.')

    # Tokenization
    parser.add_argument('--dict_type', default='core', choices=['small', 'core', 'full'],
                        help='Sudachi dictionary type to be used for tokenization.')
    parser.add_argument('--split_mode', default='C', choices=['A', 'B', 'C', 'a', 'b', 'c'],
                        help='The mode of splitting.')
    parser.add_argument('--word_form_type', default='surface',
                        choices=WordFormTypes, type=WordFormTypes,
                        help='Word form type for each morpheme.')

    # Wordpiece
    parser.add_argument('--disable_parallelism', action='store_true', default=False,
                        help='This flag argument disables parallel processing only for wordpiece training. '
                             'Note that this flag rewrites the value of a global environment variable '
                             '(TOKENIZERS_PARALLELISM), so it may affect other programs as well.')

    # Output
    parser.add_argument('-o', '--output_dir',
                        help='The output dir to be saved vocabulary and config file.')
    parser.add_argument('-c', '--config_name', default='config.json',
                        help='Output json file name.')
    parser.add_argument('-v', '--vocab_prefix', default='',
                        help='Prefix of vocab file.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
