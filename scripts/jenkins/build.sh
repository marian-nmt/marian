#!/bin/bash -ex

# Build script to be run by Jenkins.
# Should be run from the main directory (./scripts/jenkins/build.sh).

rm -rf build
mkdir build
cd build
cmake ..
make

cd ..
tar zvcf amunmt-distribution.tar.gz build/bin/* scripts/download_models.py tests/wmt16/Makefile tests/wmt16/extract_segs.py
