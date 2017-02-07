Marian
======

[![Join the chat at https://gitter.im/amunmt/marian](https://badges.gitter.im/amunmt/marian.svg)](https://gitter.im/amunmt/marian?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](http://37.247.57.181:8000/job/marian_compilation/badge/icon)](http://37.247.57.181:8000/job/marian_compilation/)

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
