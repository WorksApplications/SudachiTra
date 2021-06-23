import argparse
import os
from glob import glob

from chitra.pretokenizer import PartOfSpeechSubstitutionTokenizer


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
    parser = argparse.ArgumentParser(description='train part-of-speech substitution tokenizer')

    # input
    parser.add_argument('-f', '--input_file', default='',
                        help='input file to train tokenizer')
    parser.add_argument('-d', '--input_dir', default='',
                        help='input dir containing files to train tokenizer')

    # parameters
    parser.add_argument('--token_size', type=int, default=32000,
                        help='')
    parser.add_argument('--min_frequency', type=int, default=1,
                        help='')
    parser.add_argument('--limit_character', type=int, default=1000,
                        help='')
    parser.add_argument('--special_tokens', nargs='*', default=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                        help='')

    # sudachi
    parser.add_argument('--dict_type', default='core', choices=['small', 'core', 'full'],
                        help='')
    parser.add_argument('--split_mode', default='C', choices=['A', 'B', 'C', 'a', 'b', 'c'],
                        help='')
    parser.add_argument('--word_form_type', default='surface',
                        choices=['surface', 'dictionary', 'normalized', 'dictionary_and_surface', 'normalized_and_surface'],
                        help='')

    # output
    parser.add_argument('-o', '--output_file',
                        help='path to be saved vocab file')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
