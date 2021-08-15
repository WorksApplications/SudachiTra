from typing import Optional

from sudachipy.dictionary import Dictionary
from sudachipy.morphemelist import MorphemeList
from sudachipy.tokenizer import Tokenizer

from .tokenization_bert_sudachipy import BertSudachipyTokenizer


class SlowSudachipyWordTokenizer:
    def __init__(
            self,
            split_mode: Optional[str] = "C",
            config_path: Optional[str] = None,
            resource_dir: Optional[str] = None,
            dict_type: Optional[str] = "core",
    ):
        split_mode = split_mode.upper()
        if split_mode == "C":
            self.split_mode = Tokenizer.SplitMode.C
        elif split_mode == "B":
            self.split_mode = Tokenizer.SplitMode.B
        elif split_mode == "A":
            self.split_mode = Tokenizer.SplitMode.A
        else:
            raise ValueError("Invalid `split_mode`: " + split_mode)

        self.config_path = config_path
        self.resource_dir = resource_dir
        self.dict_type = dict_type

    def tokenize(self, text: str) -> MorphemeList:
        sudachi_dict = Dictionary(config_path=self.config_path, resource_dir=self.resource_dir, dict_type=self.dict_type)
        sudachi = sudachi_dict.create()
        return sudachi.tokenize(text, self.split_mode)


class SlowBertSudachipyTokenizer(BertSudachipyTokenizer):
    def __init__(
            self,
            vocab_file,
            do_lower_case=False,
            do_word_tokenize=True,
            do_subword_tokenize=True,
            word_tokenizer_type="sudachipy",
            subword_tokenizer_type="pos_substitution",
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
            vocab_file,
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
        self.word_tokenizer = SlowSudachipyWordTokenizer(**(self.sudachipy_kwargs or {}))

    # TODO: need to investigate the serialization behavior
    def __setstate__(self, state):
        self.__dict__ = state
        self.word_tokenizer = SlowSudachipyWordTokenizer(**(self.sudachipy_kwargs or {}))
