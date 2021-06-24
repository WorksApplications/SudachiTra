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

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from logzero import logger
from progressbar import progressbar as tqdm

from .. import SudachipyWordTokenizer
from ..tokenization_bert_sudachipy import pos_substitution_format, WORD_FORM_TYPES


class PartOfSpeechSubstitutionTokenizer(SudachipyWordTokenizer):
    def __init__(
            self,
            split_mode: Optional[str] = "C",
            dict_type: Optional[str] = "core",
            word_form_type: Optional[str] = "surface",
            **kwargs
    ):
        """
        Constructs a PartOfSpeechSubstitutionTokenizer.

        Args:
            split_mode (:obj:`str`, `optional`, defaults to :obj:`"C"`):
                The mode of splitting.
                "A", "B", or "C" can be specified.
            dict_type (:obj:`str`, `optional`, defaults to :obj:`"core"`):
                Sudachi dictionary type to be used for tokenization.
                "small", "core", or "full" can be specified.
            word_form_type (:obj:`str`, `optional`, defaults to :obj:`"surface"`):
                Word form type for each morpheme.
                "surface", "dictionary", "normalized", "dictionary_and_surface", or "normalized_and_surface" can be specified.
            **kwargs:
                Sudachi dictionary parameters.
        """
        super().__init__(split_mode=split_mode, dict_type=dict_type, **kwargs)
        self.word_form_type = word_form_type
        self.word_formatter = WORD_FORM_TYPES[self.word_form_type]
        self._vocab = None

    def get_word2freq_and_pos(self, files: List[str]) -> Tuple[Dict[str, int], List[str]]:
        """
        Tokenizes sentences in the specified files and returns tokenized data.

        Args:
            files (List[str]): List of paths of input files.

        Returns:
            Tuple[Dict[str, int], List[str]]:
                1. Dictionary of tokens and its frequency.
                2. List of part-of-speech tags.

        """
        word2freq = defaultdict(int)
        pos_list = []

        logger.info("Tokenization")
        for file in files:
            logger.info("\tReading file: {}".format(file))
            with open(file, 'r') as f:
                for line in tqdm(f):
                    line = line.strip()
                    if line != "":
                        for m in self.tokenize(line):
                            word2freq[self.word_formatter(m)] += 1
                            pos_list.append(pos_substitution_format(m))

        return word2freq, list(set(pos_list))

    def train(
            self,
            files: Union[str, List[str]],
            token_size: int = 32000,
            min_frequency: int = 1,
            limit_character: int = 1000,
            special_tokens: List[str] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    ):
        """
        Reads the input files and builds vocabulary.

        Args:
            files (str | List[str]):
                List of paths of input files to build vocabulary.
            token_size (int):
                The size of the vocabulary, excluding special tokens and pos tags.
            min_frequency (int):
                Ignores all words (tokens and characters) with total frequency lower than this.
            limit_character (int):
                The maximum different characters to keep in the vocabulary.
            special_tokens (List[str]):
                A list of special tokens the model should know of.
        """
        logger.info("Parameters for training")
        logger.info("\ttoken_size: {}".format(token_size))
        logger.info("\tmin_frequency: {}".format(min_frequency))
        logger.info("\tlimit_character: {}".format(limit_character))
        logger.info("\tspecial_tokens: {}".format(",".join(special_tokens)))

        if isinstance(files, str):
            files = [files]
        word2freq, pos_list = self.get_word2freq_and_pos(files)

        word2freq = {word: freq for word, freq in word2freq.items() if min_frequency <= freq}

        char2freq = {word: freq for word, freq in word2freq.items() if len(word) == 1}
        if limit_character < len(char2freq):
            sorted_char2freq = sorted(char2freq.items(), key=lambda x: x[1], reverse=True)
            for char, _ in sorted_char2freq[limit_character:]:
                del word2freq[char]

        word2freq = sorted(word2freq.items(), key=lambda x: x[1], reverse=True)
        if token_size < len(word2freq):
            word2freq = word2freq[:token_size]

        self._vocab = special_tokens + pos_list + list(map(lambda x: x[0], word2freq))
        logger.info("#Vocab, including POS, all (special) tokens and characters\n{}".format(len(self.vocab)))

    def save_vocab(self, output_path: str):
        """
        Saves Vocabulary into the specified path.

        Args:
            output_path (str): The output path where the vocabulary will be stored.
        """
        logger.info("Saving vocab to `{}`".format(output_path))
        with open(output_path, 'w') as f:
            f.write("\n".join(self.vocab))

    @property
    def vocab(self):
        return self._vocab
