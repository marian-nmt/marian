# Marian
-------------

[![Join the chat at https://gitter.im/amunmt/marian](https://badges.gitter.im/amunmt/marian.svg)](https://gitter.im/amunmt/marian?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](http://vali.inf.ed.ac.uk/jenkins/buildStatus/icon?job=Marian)](http://vali.inf.ed.ac.uk/jenkins/job/Marian/)

Google group for commit messages: https://groups.google.com/forum/#!forum/mariannmt

A C++ gpu-specific parallel automatic differentiation library
with operator overloading.

In honour of Marian Rejewski, a Polish mathematician and
cryptologist.

# Installation
------------

# Requirements:

* g++ with c++11
* CUDA
* Boost (>= 1.56)

# Compilation with `cmake > 3.5`:

    mkdir build
    cd build
    cmake ..
    make -j
# Acknowledgements
The development of Marian received funding from the European Union's _Horizon 2020_ Research and Innovation Programm under grant agreements 688139 ([SUMMA](http://www.summa-project.eu); 2016--2019) and 645487 ([Modern MT](http://www.modernmt.eu)).
