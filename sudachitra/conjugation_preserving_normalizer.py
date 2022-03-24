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

import json
from sudachipy.morpheme import Morpheme
from sudachipy.sudachipy import Dictionary

from .word_formatter import CONJUGATIVE_POS


class ConjugationPreservingNormalizer:
    def __init__(self, inflection_table_path, conjugation_type_table_path, sudachi_dict: Dictionary) -> None:
        """
        Constructs a ConjugationPreservingNormalizer.

        Args:
            inflection_table_path (:obj:`str`):
                In this table, get the difference between the inflected form to the end form each conjugation.
            conjugation_type_table_path (:obj:`str`):
                In this table, the conjugation types such as potential verb
                that change due to the normalization of the Sudachi dictionary are stored in blacklist format.
            sudachi_dict (:obj:`Dictionary`):
                dictionary instance.
        """
        self.sudachi_dict = sudachi_dict
        self.id2pos = list(enumerate(self.sudachi_dict.pos_matcher([()])))
        self.pos2id = {pos: pos_id for (pos_id, pos) in self.id2pos}
        self.is_necessary_inflection = sudachi_dict.pos_matcher(
            lambda p: p[0] in CONJUGATIVE_POS and (p[4] == "サ行変格" or p[5] != "終止形-一般")
        )  # 「為る->する」のため、サ行変格のみ 終止形-一般も変化

        with open(inflection_table_path) as jf:
            self.inflection_table = self._load_json(jf, "inflection")
        with open(conjugation_type_table_path) as jf:
            self.conjugation_type_table = self._load_json(jf, "conjugation_type")
    
    def _load_json(self, json_file, table_type) -> dict:
        """
        Convert json to python file when loading.

        Args:
            json_file (:obj:`TextIOWrapper`):
                json file object.
            table_type (:obj:`str`):
                Select "inflection" or "conjugation_type" and load in each format.

        Returns:
            Dict:
                inflection: Key of int and Inflection table of list.
                conjugation_type: Key of tuple and Conjugation type conversion destination of str.
        """
        if table_type not in ['inflection', 'conjugation_type']:
            raise ValueError('Invalid table_type error : {}'.format(table_type))

        data = json.load(json_file)
        table = {}
        for pos_0, convert_tables in data.items():
            for key, convert_table in convert_tables.items():
                if table_type == "inflection":
                    pos_4, pos_5 = key.split("|")
                    for pos in self.sudachi_dict.pos_matcher(lambda p: p[0] == pos_0 and p[4] == pos_4 and p[5] == pos_5):
                        pos_id = self.pos2id[pos]
                        table[pos_id] = convert_table
                elif table_type == "conjugation_type":
                    surface_token, reading, normalized_token, pos_4 = key.split("|")
                    for pos in self.sudachi_dict.pos_matcher(lambda p: p[0] == pos_0 and p[4] == pos_4):
                        pos_id = self.pos2id[pos]
                        table[(pos_id, surface_token, reading, normalized_token)] = convert_table

        return table

    def _change_pos(self, key: tuple, pos: tuple) -> tuple:
        """
        Make conjugation-type changes.

        If the part of speech does not exist after the change, it will be grouped into "一般". (Example: 撥音便, etc.)

        Args:
            key (:obj:`tuple`): (pos_id, surface, reading_form, normalized_form)
            pos (:obj:`tuple`): Get the part of speech.

        Returns:
            tuple: Changed part of speech
        """
        conj_type = self.conjugation_type_table[key]
        res = (*pos[:4], conj_type, pos[5])
        if res not in self.pos2id:
            conj_form = pos[5].split("-")[0] + "-一般"
            res = (*pos[:4], conj_type, conj_form)
            if res not in self.pos2id:
                res = (pos[0], "一般", pos[2], pos[3], conj_type, conj_form)

        assert res in self.pos2id
        return res
    
    def _is_changed_conjugation_type(self, key: tuple) -> bool:
        """
        Check if the conjugation type changes after being normalized.

        Args:
            key (:obj:`tuple`): (pos_id, surface, reading_form, normalized_form)

        Returns: bool
        """
        return key in self.conjugation_type_table

    def normalized(self, morpheme: Morpheme) -> str:
        """
        The output token retain conjugation information in word normalization by Sudachi tokenizer

        Args:
            morpheme (:obj:`Morpheme`): A Morpheme obtained from the analysis by sudachipy.

        Returns:
            str: Normalized token with conjugate information retained.
        """
        normalized_token = morpheme.normalized_form()
        if not self.is_necessary_inflection(morpheme):
            return normalized_token

        pos_id = morpheme.part_of_speech_id()
        conj_type_table_key = (pos_id, morpheme.surface(), morpheme.reading_form(), normalized_token)
        if self._is_changed_conjugation_type(conj_type_table_key):
            pos = self._change_pos(conj_type_table_key, morpheme.part_of_speech())
            if pos[5] == "終止形-一般":
                return normalized_token
            pos_id = self.pos2id[pos]

        for convert_table in self.inflection_table[pos_id]:
            if convert_table == ['', '']:
                return normalized_token
            if normalized_token.endswith(convert_table[0]):
                src = convert_table[0][::-1]
                tgt = convert_table[1][::-1]
                return normalized_token[::-1].replace(src, tgt, 1)[::-1]

        return normalized_token
