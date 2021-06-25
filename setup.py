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
    packages=find_packages(exclude=['tests']),
    install_requires=[
        "transformers>=4.6.1",
        "sudachipy>=0.5.2",
        "sudachidict_core>=20210608"
    ],
    include_package_data=True
)
