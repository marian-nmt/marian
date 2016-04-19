
# amuNN

A C++ decoder for Neural Machine Translation (NMT) models trained with Theano-based scripts from DL4MT (https://github.com/nyu-dl/dl4mt-tutorial)

## Requirements:
 * CMake 3.5.1
 * Boost 1.5
 * CUDA 7.5
 * KenLM (https://github.com/kpu/kenlm, current master)

## Compilation
The project is a standard Cmake out-of-source build:

    mkdir build
    cd build
    cmake .. -DKENLM=path/to/kenlm
    make -j

On Ubuntu 16.04, you need g++4.9 and cuda-7.5 and a boost version compiled with g++4.9

    CUDA_BIN_PATH=/usr/local/cuda-7.5 BOOST_ROOT=/home/marcin/myboost cmake .. \
    -DCMAKE_CXX_COMPILER=g++-4.9 -DCUDA_HOST_COMPILER=/usr/bin/g++-4.9

Vocabularies (*.pkl extension) need to be converted to text with the scripts in the scripts folder.

    python scripts/vocab2txt.py vocab.en.pkl > vocab.en
