Marian
======

[![Join the chat at https://gitter.im/MarianNMT/Lobby](https://badges.gitter.im/MarianNMT/Lobby.svg)](https://gitter.im/MarianNMT/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Google group for commit messages: https://groups.google.com/forum/#!forum/mariannmt

A C++ gpu-specific parallel automatic differentiation library
with operator overloading.

In honour of Marian Rejewski, a Polish mathematician and
cryptologist.

Installation
------------

Requirements:

* g++ with c++11
* CUDA
* Boost (>= 1.56)

Compilation with `cmake > 3.5`:

    mkdir build
    cd build
    cmake ..
    make -j

To compile API documentation using Doxygen, first cd to the build directory, and then:

    make doc

To test, first compile, then:

    cd examples/mnist
    make
    cd ../../build
    ./mnist_benchmark
