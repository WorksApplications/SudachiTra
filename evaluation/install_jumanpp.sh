#!/bin/bash

set -eu

# install jumanpp v2.0.0-rc3
# ref: https://qiita.com/Gushi_maru/items/ee434b5bc9f020c8feb6

if [ ! -d ./jumanpp-2.0.0-rc3 ]; then
    wget https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz
    tar -xvf jumanpp-2.0.0-rc3.tar.xz
fi
if [ ! -d jumanpp-2.0.0-rc3/build ]; then
    mkdir jumanpp-2.0.0-rc3/build
fi

cd jumanpp-2.0.0-rc3/build/

apt install cmake -y

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local/usr
make
make install

