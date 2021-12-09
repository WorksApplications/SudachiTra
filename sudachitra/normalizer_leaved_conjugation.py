
import json
from sudachipy.morpheme import Morpheme

from sudachitra.tokenization_bert_sudachipy import CONJUGATIVE_POS


class NormalizerLeavedConjugation:
    def __init__(self, inflectoin_table_path, conjugation_type_table_path) -> None:
        with open(inflectoin_table_path) as jf:
            self.infl_table = json.load(jf)
        with open(conjugation_type_table_path) as jf:
            self.conj_type_table = json.load(jf)
        self.errlog = open("err.log", "w")

    def _read_pos(self, morpheme: Morpheme) -> list:
        pos = list(morpheme.part_of_speech())
        key = "|".join((morpheme.surface(), morpheme.reading_form(),
                       morpheme.normalized_form(), pos[4]))
        if key in self.conj_type_table[pos[0]]:
            pos[4] = self.conj_type_table[pos[0]][key]
            if pos[4]+"|"+pos[5] not in self.infl_table[pos[0]]:
                pos[5] = pos[5].split("-")[0] + "-一般"
        return pos

    def normalized(self, morpheme: Morpheme) -> str:
        normalized_token = morpheme.normalized_form()
        if morpheme.part_of_speech()[0] not in CONJUGATIVE_POS:
            return normalized_token

        pos = self._read_pos(morpheme)
        if pos[5] != '終止形-一般' or pos[4] == "サ行変格":  # 「為る->する」のため、サ行変格のみ 終止形-一般も変化
            convert_tables = self.infl_table[pos[0]][pos[4]+"|"+pos[5]]
            for convert_table in convert_tables:
                if convert_table == ['', '']:
                    return normalized_token
                if normalized_token.endswith(convert_table[0]):
                    src = convert_table[0][::-1]
                    tgt = convert_table[1][::-1]
                    return normalized_token[::-1].replace(src, tgt, 1)[::-1]

        return normalized_token



