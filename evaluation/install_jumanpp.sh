#!/bin/bash

set -eu

# install jumanpp v2.0.0-rc3
# ref: https://qiita.com/Gushi_maru/items/ee434b5bc9f020c8feb6
wget https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz
tar -xvf jumanpp-2.0.0-rc3.tar.xz

apt install cmake -y

cd jumanpp-2.0.0-rc3/
mkdir build
cd build/

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
make
make install

