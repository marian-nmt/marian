
# amuNN

## Requirements:
 * CMake
 * Boost
 * CUDA
 * KenLM

## Compilation
The project is a standard Cmake project:

    mkdir build
    cd build
    cmake ..
    make -j

## KenLM
To compile amuNN you need the lastest version of [kenLM](https://github.com/kpu/kenlm) and pass the path as a Cmake variable KENLM, e.g. :

    cmake -DKENLM=../kenlm ..

On Ubuntu 16.04, you need g++4.9 and cuda-7.5 and a boost version compiled with g++4.9

    CUDA_BIN_PATH=/usr/local/cuda-7.5 BOOST_ROOT=/home/marcin/myboost cmake .. -DCMAKE_CXX_COMPILER=g++-4.9 -DCUDA_HOST_COMPILER=/usr/bin/g++-4.9

Vocabularies need to be converted to text with the scripts in the scripts folder.
