#!/bin/bash -ex

# Build script to be run by Jenkins.
# Should be run from the main directory (./scripts/jenkins/build.sh).

rm -rf build
mkdir build
cd build
cmake ..
make
