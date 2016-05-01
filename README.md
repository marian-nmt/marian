
# amuNN

A C++ decoder for Neural Machine Translation (NMT) models trained with Theano-based scripts from 
Nematus (https://github.com/rsennrich/nematus) or DL4MT (https://github.com/nyu-dl/dl4mt-tutorial)

## Requirements:
 * CMake 3.5.1 (due to CUDA related bugs in earlier versions)
 * Boost 1.5
 * CUDA 7.5
 * yaml-cpp 0.5 (https://github.com/jbeder/yaml-cpp.git)

## Optional
 * KenLM for n-gram language models (https://github.com/kpu/kenlm, current master)

## Compilation
The project is a standard Cmake out-of-source build:

    mkdir build
    cd build
    cmake ..
    make -j

Or with KenLM support:

    cmake .. -DKENLM=path/to/kenlm


On Ubuntu 16.04, you currently need g++4.9 to compile and cuda-7.5, this also requires a custom boost build compiled with g++4.9 instead of the standard g++5.3. The binaries are not compatible. g++5 support will probably arrive with cuda-8.0.

    CUDA_BIN_PATH=/usr/local/cuda-7.5 BOOST_ROOT=/path/to/custom/boost cmake .. \
    -DCMAKE_CXX_COMPILER=g++-4.9 -DCUDA_HOST_COMPILER=/usr/bin/g++-4.9

## Vocabulary files
Vocabulary files (and all other config files) in amuNN are by default YAML files. AmuNN also reads gzipped yml.gz files. 

* Vocabularies from the DL4MT repository (*.pkl extension) need to be converted to JSON/YAML:
```    
python scripts/vocab2yaml.py vocab.en.pkl > vocab.en</blockquote>
```
* Vocabulary files from Nematus can be used directly, as JSON is a proper subset of YAML. 
