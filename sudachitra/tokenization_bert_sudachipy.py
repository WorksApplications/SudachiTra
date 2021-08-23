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

import copy
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from sudachipy.morpheme import Morpheme
from transformers.models.bert_japanese.tokenization_bert_japanese import CharacterTokenizer
from transformers.models.bert.tokenization_bert import WordpieceTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from .sudachipy_word_tokenizer import SudachipyWordTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# TODO: set official URL
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "WorksApplications/bert-base-japanese-sudachi": "https://.../vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "WorksApplications/bert-base-japanese-sudachi": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "WorksApplications/bert-base-japanese-sudachi": {
        "do_lower_case": False,
        "word_tokenizer_type": "sudachipy",
        "subword_tokenizer_type": "pos_substitution",
    },
}


def load_vocabulary(vocab_file: str = VOCAB_FILES_NAMES["vocab_file"]) -> Dict[str, int]:
    """
    Loads a vocabulary file into a dictionary.

    Args:
        vocab_file (str): Vocabulary file path.

    Returns:
        Dict[str, int]: Dictionary of vocabulary and its index.
    """
    vocab = OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n").split("\t")[0]
        vocab[token] = index
    return vocab


def save_vocabulary(vocab: Dict[str, int], save_directory: str, filename_prefix: Optional[str] = None,
                    vocab_file: str = VOCAB_FILES_NAMES["vocab_file"]) -> Tuple[str]:
    """
    Save the vocabulary.

    Args:
        vocab (Dict[str, int]): Dictionary of vocabulary and its index.
        save_directory (str): The output directory where the vocabulary will be stored.
        filename_prefix (str | None): The filename prefix of the vocabulary file.
        vocab_file (str): The filename of the vocabulary file.

    Returns:
        Tuple[str]: Saved vocabulary path.
    """
    index = 0
    if os.path.isdir(save_directory):
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + vocab_file
        )
    else:
        vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
    with open(vocab_file, "w", encoding="utf-8") as writer:
        for token, token_index in sorted(vocab.items(), key=lambda kv: kv[1]):
            assert index == token_index, f"Error {vocab_file}: vocabulary indices are not consecutive, '{token}' {index} != {token_index}."
            writer.write(token + "\n")
            index += 1
    return (vocab_file,)


SUBWORD_TOKENIZER_TYPES = [
    "pos_substitution",
    "wordpiece",
    "character",
]

HALF_ASCII_TRANSLATE_TABLE = str.maketrans({chr(0xFF01 + _): chr(0x21 + _) for _ in range(94)})

WORD_FORM_TYPES = {
    "surface": lambda m: m.surface(),
    "dictionary": lambda m: m.dictionary_form(),
    "normalized": lambda m: m.normalized_form(),
    "dictionary_and_surface": lambda m: m.surface() if m.part_of_speech()[0] in CONJUGATIVE_POS else m.dictionary_form(),
    "normalized_and_surface": lambda m: m.surface() if m.part_of_speech()[0] in CONJUGATIVE_POS else m.normalized_form(),
    "surface_half_ascii": lambda m: m.surface().translate(HALF_ASCII_TRANSLATE_TABLE),
    "dictionary_half_ascii": lambda m: m.dictionary_form().translate(HALF_ASCII_TRANSLATE_TABLE),
    "dictionary_and_surface_half_ascii": lambda m: m.surface().translate(HALF_ASCII_TRANSLATE_TABLE) if m.part_of_speech()[0] in CONJUGATIVE_POS else m.dictionary_form().translate(HALF_ASCII_TRANSLATE_TABLE),
}

CONJUGATIVE_POS = {'動詞', '形容詞', '形容動詞', '助動詞'}


def pos_substitution_format(token: Morpheme) -> str:
    """
    Creates POS tag by combining each POS.

    Args:
        token (Morpheme): Morpheme in a sentence.

    Returns:
        str: POS tag.
    """
    hierarchy = token.part_of_speech()
    pos = [hierarchy[0]]
    for p in hierarchy[1:]:
        if p == "*":
            break
        pos.append(p)

    return "[{}]".format("-".join(pos))


class BertSudachipyTokenizer(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
            self,
            vocab_file,
            do_lower_case=False,
            do_word_tokenize=True,
            do_subword_tokenize=True,
            word_tokenizer_type="sudachipy",
            subword_tokenizer_type="wordpiece",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            word_form_type="surface",
            sudachipy_kwargs=None,
            **kwargs
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            do_word_tokenize=do_word_tokenize,
            do_subword_tokenize=do_subword_tokenize,
            word_tokenizer_type=word_tokenizer_type,
            subword_tokenizer_type=subword_tokenizer_type,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            word_form_type=word_form_type,
            sudachipy_kwargs=sudachipy_kwargs,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'.")

        self.vocab = load_vocabulary(vocab_file)
        self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        if not do_word_tokenize:
            raise ValueError(f"`do_word_tokenize` must be True.")
        elif word_tokenizer_type != "sudachipy":
            raise ValueError(f"Invalid word_tokenizer_type '{word_tokenizer_type}' is specified.")

        self.lower_case = do_lower_case

        self.sudachipy_kwargs = copy.deepcopy(sudachipy_kwargs)
        self.word_tokenizer = SudachipyWordTokenizer(**(self.sudachipy_kwargs or {}))
        self.word_form_type = word_form_type

        self.do_subword_tokenize = do_subword_tokenize
        self.subword_tokenizer_type = subword_tokenizer_type
        if do_subword_tokenize:
            if subword_tokenizer_type == "pos_substitution":
                self.subword_tokenizer = None
            elif subword_tokenizer_type == "wordpiece":
                self.subword_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)
            elif subword_tokenizer_type == "character":
                self.subword_tokenizer = CharacterTokenizer(vocab=self.vocab, unk_token=self.unk_token)
            else:
                raise ValueError(f"Invalid subword_tokenizer_type '{subword_tokenizer_type}' is specified.")

    @property
    def do_lower_case(self):
        return self.lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # TODO: need to investigate the serialization behavior
    def __getstate__(self):
        state = dict(self.__dict__)
        del state["word_tokenizer"]
        return state

    # TODO: need to investigate the serialization behavior
    def __setstate__(self, state):
        self.__dict__ = state
        self.word_tokenizer = SudachipyWordTokenizer(**(self.sudachipy_kwargs or {}))

    def _tokenize(self, text, **kwargs):
        tokens = self.word_tokenizer.tokenize(text)
        word_format = WORD_FORM_TYPES[self.word_form_type]
        if self.do_subword_tokenize:
            if self.subword_tokenizer_type == "pos_substitution":
                def _substitution(token):
                    word = word_format(token)
                    if word in self.vocab:
                        return word
                    substitute = pos_substitution_format(token)
                    if substitute in self.vocab:
                        return substitute
                    return self.unk_token
                split_tokens = [_substitution(token) for token in tokens]
            else:
                split_tokens = [sub_token for token in tokens for sub_token in self.subword_tokenizer.tokenize(
                    word_format(token)
                )]
        else:
            split_tokens = [word_format(token) for token in tokens]

        return split_tokens

    # <-- ported from BertTokenizer

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # ported from BertTokenizer -->

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None,
                        vocab_file_name: str = VOCAB_FILES_NAMES["vocab_file"]) -> Tuple[str]:
        """
        Save the vocabulary.

        Args:
            save_directory (str): The output directory where the vocabulary will be stored.
            filename_prefix (str | None): The filename prefix of the vocabulary file.
            vocab_file_name (str): The filename of the vocabulary file.

        Returns:
            Tuple[str]: Saved vocabulary path.
        """
        return save_vocabulary(self.vocab, save_directory, filename_prefix, vocab_file_name)
