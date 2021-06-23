from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from logzero import logger
from progressbar import progressbar as tqdm

from chitra import SudachipyWordTokenizer
from chitra.tokenization_bert_sudachipy import pos_substitution_format, WORD_FORM_TYPES


class PartOfSpeechSubstitutionTokenizer(SudachipyWordTokenizer):
    def __init__(
            self,
            split_mode: Optional[str] = "C",
            dict_type: Optional[str] = "core",
            word_form_type: Optional[str] = "surface",
            **kwargs
    ):
        super().__init__(split_mode=split_mode, dict_type=dict_type, **kwargs)
        self.word_form_type = word_form_type
        self.word_formatter = WORD_FORM_TYPES[self.word_form_type]
        self._vocab = None

    def get_word2freq_and_pos(self, files: List[str]) -> Tuple[Dict[str, int], List[str]]:
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
        logger.info("Parameters for training")
        logger.info("\ttoken_size: {}".format(token_size))
        logger.info("\tmin_frequency: {}".format(min_frequency))
        logger.info("\tlimit_character: {}".format(limit_character))
        logger.info("\tspecial_tokens: {}".format(",".join(special_tokens)))

        if isinstance(files, str):
            files = [files]
        word2freq, pos_list = self.get_word2freq_and_pos(files)

        # cut low frequently words
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
        logger.info("Saving vocab to `{}`".format(output_path))
        with open(output_path, 'w') as f:
            f.write("\n".join(self.vocab))

    @property
    def vocab(self):
        return self._vocab
