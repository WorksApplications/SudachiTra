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

from . import BertSudachipyTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# TODO: set official URL
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "megagonlabs/electra-base-ud-japanese": "https://.../vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "megagonlabs/electra-base-ud-japanese": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "megagonlabs/electra-base-ud-japanese": {
        "do_lower_case": False,
        "word_tokenizer_type": "sudachipy",
        "subword_tokenizer_type": "pos_substitution",
    },
}


class ElectraSudachipyTokenizer(BertSudachipyTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
