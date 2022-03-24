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

from tokenizers.normalizers import Lowercase, NFKC, Sequence, Strip


class InputStringNormalizer(object):
    def __init__(self, do_strip=False, do_lower_case=False, do_nfkc=False):
        self.do_strip: bool = do_strip
        self.do_lower_case: bool = do_lower_case
        self.do_nfkc: bool = do_nfkc
        self._normalizer: Sequence = self._init_normalizer()

    def _init_normalizer(self) -> Sequence:
        normalizers = []
        if self.do_strip:
            normalizers.append(Strip())
        if self.do_lower_case:
            normalizers.append(Lowercase())
        if self.do_nfkc:
            normalizers.append(NFKC())
        return Sequence(normalizers)

    @property
    def normalizer(self) -> Sequence:
        return self._normalizer

    def normalize_str(self, text: str) -> str:
        return self.normalizer.normalize_str(text)
