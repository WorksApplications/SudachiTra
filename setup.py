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

from setuptools import find_packages, setup

setup(
    name="SudachiTra",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Japanese tokenizer for Transformers.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/WorksApplications/SudachiTra",
    license="Apache-2.0",
    author="Works Applications",
    author_email="sudachi@worksap.co.jp",
    packages=find_packages(include=['sudachitra']),
    install_requires=[
        "logzero~=1.7.0",
        "tokenizers>=0.10.3",
        "transformers>=4.6.1",
        "sudachipy>=0.6.2",
        "sudachidict_core>=20210802"
    ],
    extras_require={
        "pretrain":[
            "progressbar2~=3.53.1",
            "bunkai~=1.4.3",
            "pytextspan>=0.5.4",
            "tensorflow>=2.5.0",
            "tensorflow_datasets>=4.3.0"
        ]
    },
    include_package_data=True
)
