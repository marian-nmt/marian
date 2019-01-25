#!/bin/bash -v

mkdir -p bin

# download and compile fast_align
if [ ! -e bin/fast_align ]; then
    git clone https://github.com/clab/fast_align
    mkdir -p fast_align/build
    cd fast_align/build
    cmake ..
    make -j4
    cp fast_align atools ../../bin
    cd ../../
fi

# download and compile extract-lex
if [ ! -e bin/extract_lex ]; then
    git clone https://github.com/marian-nmt/extract-lex
    mkdir -p extract-lex/build
    cd extract-lex/build
    cmake ..
    make -j4
    cp extract_lex ../../bin
    cd ../../
fi
