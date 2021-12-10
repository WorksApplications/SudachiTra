
import json
from sudachipy.morpheme import Morpheme
from sudachipy.sudachipy import Dictionary

from sudachitra.tokenization_bert_sudachipy import CONJUGATIVE_POS


class NormalizerLeavedConjugation:
    def __init__(self, inflection_table_path, conjugation_type_table_path, sudachi_dict: Dictionary) -> None:
        self.sudachi_dict = sudachi_dict
        self.is_conjugative_pos = sudachi_dict.pos_matcher(lambda p: p[0] in CONJUGATIVE_POS)
        self.is_needs_inflection = sudachi_dict.pos_matcher(lambda p: p[4] == "サ行変格" or p[5] != "終止形-一般") # 「為る->する」のため、サ行変格のみ 終止形-一般も変化

        with open(inflection_table_path) as jf:
            self.infl_table = self._load_json(jf)
        with open(conjugation_type_table_path) as jf:
            self.conj_type_table = self._load_json(jf)
    
    def _load_json(self, json_file) -> dict:
        data = json.load(json_file)
        table = {(pos_0, *key.split("|")): convert_table for pos_0, convert_tables in data.items() for key, convert_table in convert_tables.items()}
        return table

    def _change_pos(self, key, pos) -> tuple:
        conj_type = self.conj_type_table[key]
        conj_form = pos[5]
        if (pos[0], conj_type, conj_form) not in self.infl_table:
            conj_form = pos[5].split("-")[0] + "-一般"
        return (*pos[:4], conj_type, conj_form)
    
    def _is_changed_conjugation_type(self, key) -> bool:
        return key in self.conj_type_table

    def normalized(self, morpheme: Morpheme) -> str:
        normalized_token = morpheme.normalized_form()
        if not self.is_conjugative_pos(morpheme) or not self.is_needs_inflection(morpheme):
            return normalized_token

        pos = morpheme.part_of_speech()
        conj_type_table_key = (pos[0], morpheme.surface(), morpheme.reading_form(), normalized_token, pos[4])
        if self._is_changed_conjugation_type(conj_type_table_key):
            pos = self._change_pos(conj_type_table_key, pos)
            if pos[5] != "終止形-一般":
                return normalized_token

        for convert_table in self.infl_table[(pos[0], pos[4], pos[5])]:
            if convert_table == ['', '']:
                return normalized_token
            if normalized_token.endswith(convert_table[0]):
                src = convert_table[0][::-1]
                tgt = convert_table[1][::-1]
                return normalized_token[::-1].replace(src, tgt, 1)[::-1]

        return normalized_token



