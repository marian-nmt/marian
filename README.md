
# AmuNMT

A C++ decoder for Neural Machine Translation (NMT) models trained with Theano-based scripts from 
Nematus (https://github.com/rsennrich/nematus) or DL4MT (https://github.com/nyu-dl/dl4mt-tutorial)

We aim at keeping compatibility with Nematus (at least as long as there is no training framework in AmunNMT), the continued compatbility with DL4MT will not be guaranteed. 

## Requirements:
 * CMake 3.5.1 (due to CUDA related bugs in earlier versions)
 * Boost 1.5
 * CUDA 7.5

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
Vocabulary files (and all other config files) in AmuNMT are by default YAML files. AmuNMT also reads gzipped yml.gz files. 

* Vocabulary files from models trained with Nematus can be used directly as JSON is a proper subset of YAML.
* Vocabularies for models trained with DL4MT (*.pkl extension) need to be converted to JSON/YAML with either of the two scripts below:
```    
python scripts/pkl2json.py vocab.en.pkl > vocab.json
python scripts/pkl2yaml.py vocab.en.pkl > vocab.yml
``` 

## Running AmuNMT

    ./bin/amun -c config.yml <<< "This is a test ."

## Configuration files

An example configuration:

    # Paths are relative to config file location
    relative-paths: yes

    # performance settings
    beam-size: 12
    devices: [0]
    normalize: yes
    threads-per-device: 1
    
    # scorer configuration
    scorers: 
      F0:
        path: model.en-de.npz 
        type: Nematus

    # scorer weights
    weights: 
      F0: 1.0
  
    # vocabularies
    source-vocab: vocab.en.yml.gz
    target-vocab: vocab.de.yml.gz
    
