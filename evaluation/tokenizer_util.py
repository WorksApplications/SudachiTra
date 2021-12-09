import fugashi
import mojimoji
from pyknp import Juman


class MecabJuman():
    def __init__(self, dicdir=None, mecabrc=None):
        # assume existance of followings (installed by `apt install mecab`)
        dicdir = dicdir or "/var/lib/mecab/dic/juman-utf8"
        mecabrc = mecabrc or "/etc/mecabrc"
        assert dicdir and mecabrc

        tagger = fugashi.GenericTagger(f"-r {mecabrc} -d {dicdir}")
        charset = tagger.dictionary_info[0]["charset"]
        assert charset in ["utf-8", "utf8"]

        self.tagger = tagger
        return

    def tokenize(self, line) -> str:
        # tokenize text and
        normalized = mojimoji.han_to_zen(line).replace("\u3000", " ")
        tokens = []
        for w in self.tagger(normalized):
            try:
                tokens.append(w.surface)
            except:
                pass
        return " ".join(tokens)

    def __call__(self, line) -> str:
        return self.tokenize(line)


class Juman():
    def __init__(self):
        # assume Juman++ is installed (see install_jumanpp.sh)
        self.tok = Juman()
        return

    def tokenize(self, line) -> str:
        normalized = mojimoji.han_to_zen(line).replace("\u3000", " ")
        morphs = self.tok.analysis(normalized)
        return " ".join(m.midasi for m in morphs)

    def __call__(self, line) -> str:
        return self.tokenize(line)
