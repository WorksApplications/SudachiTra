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

import os
from typing import Dict, Iterator, List, Optional, TypeVar, Union

from logzero import logger
from tokenizers import Tokenizer, AddedToken, decoders, trainers
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer
from tokenizers.processors import BertProcessing
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.implementations.base_tokenizer import BaseTokenizer

from .sudachipy_pretokenizer import CustomPreTokenizer


CPT = TypeVar('CPT', bound=CustomPreTokenizer)


class JapaneseBertWordPieceTokenizer(BaseTokenizer):
    def __init__(
            self,
            vocab: Optional[Union[str, Dict[str, int]]] = None,
            unk_token: Union[str, AddedToken] = "[UNK]",
            sep_token: Union[str, AddedToken] = "[SEP]",
            cls_token: Union[str, AddedToken] = "[CLS]",
            pad_token: Union[str, AddedToken] = "[PAD]",
            mask_token: Union[str, AddedToken] = "[MASK]",
            lowercase: bool = False,
            wordpieces_prefix: str = "##",
    ):
        if vocab is not None:
            tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(unk_token)))
        else:
            tokenizer = Tokenizer(WordPiece(unk_token=str(unk_token)))

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(sep_token)) is not None:
            tokenizer.add_special_tokens([str(sep_token)])
        if tokenizer.token_to_id(str(cls_token)) is not None:
            tokenizer.add_special_tokens([str(cls_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])

        tokenizer.normalizer = Sequence([Lowercase(), NFKC()])
        tokenizer.pre_tokenizer = BertPreTokenizer()

        if vocab is not None:
            sep_token_id = tokenizer.token_to_id(str(sep_token))
            if sep_token_id is None:
                raise TypeError("sep_token not found in the vocabulary")
            cls_token_id = tokenizer.token_to_id(str(cls_token))
            if cls_token_id is None:
                raise TypeError("cls_token not found in the vocabulary")

            tokenizer.post_processor = BertProcessing(
                (str(sep_token), sep_token_id), (str(cls_token), cls_token_id)
            )
        tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)

        parameters = {
            "model": "BertSudachiWordPiece",
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "lowercase": lowercase,
            "wordpieces_prefix": wordpieces_prefix,
        }

        super().__init__(tokenizer, parameters)

    @staticmethod
    def from_file(vocab: str, **kwargs):
        vocab = WordPiece.read_file(vocab)
        return BertWordPieceTokenizer(vocab, **kwargs)

    def train(
            self,
            files: Union[str, List[str]],
            vocab_size: int = 30000,
            min_frequency: int = 2,
            limit_alphabet: int = 1000,
            initial_alphabet: List[str] = [],
            special_tokens: List[Union[str, AddedToken]] = [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
            ],
            show_progress: bool = True,
            wordpieces_prefix: str = "##",
    ):
        """ Train the model using the given files """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        logger.info("Parameters for training")
        logger.info("\tvocab_size: {}".format(vocab_size))
        logger.info("\tmin_frequency: {}".format(min_frequency))
        logger.info("\tlimit_alphabet: {}".format(limit_alphabet))
        logger.info("\tinitial_alphabet: {}".format(",".join(initial_alphabet)))
        logger.info("\tspecial_tokens: {}".format(",".join(special_tokens)))
        logger.info("\twordpieces_prefix: {}".format(wordpieces_prefix))

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            special_tokens=special_tokens,
            show_progress=show_progress,
            continuing_subword_prefix=wordpieces_prefix,
        )
        if isinstance(files, str):
            files = [files]

        logger.info("Input files")
        logger.info("\n".join(map(lambda x: "\t{}".format(x), files)))

        self._tokenizer.train(files, trainer=trainer)
        logger.info("#Vocab: {}".format(self.get_vocab_size()))

    def train_from_iterator(
            self,
            iterator: Union[Iterator[str], Iterator[Iterator[str]]],
            vocab_size: int = 30000,
            min_frequency: int = 2,
            limit_alphabet: int = 1000,
            initial_alphabet: List[str] = [],
            special_tokens: List[Union[str, AddedToken]] = [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
            ],
            show_progress: bool = True,
            wordpieces_prefix: str = "##",
    ):
        """ Train the model using the given iterator """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        logger.info("Parameters for training")
        logger.info("\tvocab_size: {}".format(vocab_size))
        logger.info("\tmin_frequency: {}".format(min_frequency))
        logger.info("\tlimit_alphabet: {}".format(limit_alphabet))
        logger.info("\tinitial_alphabet: {}".format(",".join(initial_alphabet)))
        logger.info("\tspecial_tokens: {}".format(",".join(special_tokens)))
        logger.info("\twordpieces_prefix: {}".format(wordpieces_prefix))

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            special_tokens=special_tokens,
            show_progress=show_progress,
            continuing_subword_prefix=wordpieces_prefix,
        )
        self._tokenizer.train_from_iterator(iterator, trainer=trainer)

        logger.info("#Vocab: {}".format(self.get_vocab_size()))

    def set_pre_tokenizer(self, custom_pre_tokenizer: CPT):
        """
        Sets the custom tokenizer as pre-tokenizer.

        Args:
            custom_pre_tokenizer (CPT): Custom tokenizer that implements `custom_split`.
        """
        self.pre_tokenizer = PreTokenizer.custom(custom_pre_tokenizer)

    def save(self, output_tokenizer_path: str, pretty: bool = False):
        """
        Saves a config file of the tokenizer.

        Notes: To serialize tokenizer, puts dummy pre-tokenizer.

        Args:
            output_tokenizer_path (str): Output file path to be saved a config file of tokenizer.
            pretty (bool): Json format type.
        """
        logger.info("Saving config to `{}`".format(output_tokenizer_path))

        self.pre_tokenizer = BertPreTokenizer()  # dummy
        super().save(output_tokenizer_path, pretty=pretty)

    def save_vocab(self, output_dir: str, prefix: str):
        """
        Save the vocabulary.

        Args:
            output_dir (str): The path to the target directory in which to save the various files.
            prefix (str): An optional prefix, used to prefix each file name
        """
        logger.info("Saving vocab to `{}`".format(os.path.join(output_dir, "{}-vocab.txt".format(prefix))))

        self.model.save(output_dir, prefix)
