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

from typing import Optional

from sudachipy import Dictionary
from sudachipy import MorphemeList
from sudachipy import SplitMode


def get_split_mode(split_mode: str) -> SplitMode:
    """
    Returns the specified SplitMode.
    "A", "B", or "C" can be specified.

    Args:
        split_mode (str): The mode of splitting.

    Returns:
        SplitMode: Unit to split text.

    Raises:
        ValueError: If `split_mode` is not defined in SudachiPy.
    """
    split_mode = split_mode.upper()
    if split_mode == "C":
        return SplitMode.C
    elif split_mode == "B":
        return SplitMode.B
    elif split_mode == "A":
        return SplitMode.A
    else:
        raise ValueError("Invalid `split_mode`: " + split_mode)


class SudachipyWordTokenizer:
    """Runs tokenization with SudachiPy."""

    def __init__(
            self,
            split_mode: Optional[str] = "A",
            config_path: Optional[str] = None,
            resource_dir: Optional[str] = None,
            dict_type: Optional[str] = "core",
    ):
        """
        Constructs a SudachipyTokenizer.

        Args:
            split_mode (:obj:`str`, `optional`, defaults to :obj:`"C"`):
                The mode of splitting.
                "A", "B", or "C" can be specified.
            config_path (:obj:`str`, `optional`, defaults to :obj:`None`):
                Path to a config file of SudachiPy to be used for the sudachi dictionary initialization.
            resource_dir (:obj:`str`, `optional`, defaults to :obj:`None`):
                Path to a resource dir containing resource files, such as "sudachi.json".
            dict_type (:obj:`str`, `optional`, defaults to :obj:`"core"`):
                Sudachi dictionary type to be used for tokenization.
                "small", "core", or "full" can be specified.
        """
        self.split_mode = get_split_mode(split_mode)

        self.sudachi_dict = Dictionary(config_path=config_path, resource_dir=resource_dir, dict=dict_type)
        self.sudachi = self.sudachi_dict.create(self.split_mode)

    def tokenize(self, text: str) -> MorphemeList:
        """
        Tokenizes the specified text and returns its morphemes.

        Args:
            text (str): Input string.

        Returns:
            MorphemeList: List of morphemes.
        """
        return self.sudachi.tokenize(text)
