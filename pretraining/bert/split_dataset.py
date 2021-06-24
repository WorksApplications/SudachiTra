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

import argparse
import os
from progressbar import progressbar as tqdm
from typing import List


def write_lines(tmp_lines: List[str], file_dir: str, file_name: str, file_id: int, file_ext: str):
    with open(os.path.join(file_dir, f"{file_name}{file_id:0=2}{file_ext}"), 'w') as f:
        for line in tmp_lines:
            print(line, file=f)


def main():
    args = get_args()

    file_dir = os.path.dirname(args.input_file)
    file_name, file_ext = os.path.splitext(os.path.basename(args.input_file))

    with open(args.input_file, 'r') as f:
        lines = f.readlines()

    tmp_lines = []
    file_id = 1
    for line in tqdm(lines):
        tmp_lines.append(line.strip())
        if len(tmp_lines) > args.line_per_file and line == '\n':
            write_lines(tmp_lines, file_dir, file_name, file_id, file_ext)
            file_id += 1
            tmp_lines = []

    if len(tmp_lines) > 0:
        write_lines(tmp_lines, file_dir, file_name, file_id, file_ext)


def get_args():
    parser = argparse.ArgumentParser(description='Split dataset.')

    parser.add_argument('-i', '--input_file', help='Input file to be splitted (corpus splitted by paragraph).')
    parser.add_argument('--line_per_file', help='Max number of lines per file.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
