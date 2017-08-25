fast_align
==========

`fast_align` is a simple, fast, unsupervised word aligner.

If you use this software, please cite:
* [Chris Dyer](http://www.cs.cmu.edu/~cdyer), [Victor Chahuneau](http://victor.chahuneau.fr), and [Noah A. Smith](http://www.cs.cmu.edu/~nasmith). (2013). [A Simple, Fast, and Effective Reparameterization of IBM Model 2](http://www.ark.cs.cmu.edu/cdyer/fast_valign.pdf). In *Proc. of NAACL*.

The source code in this repository is provided under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).

## Input format

Input to `fast_align` must be tokenized and aligned into parallel sentences. Each line is a source language sentence and its target language translation, separated by a triple pipe symbol with leading and trailing white space (` ||| `). An example 3-sentence German–English parallel corpus is:

    doch jetzt ist der Held gefallen . ||| but now the hero has fallen .
    neue Modelle werden erprobt . ||| new models are being tested .
    doch fehlen uns neue Ressourcen . ||| but we lack new resources .

## Compiling and using `fast_align`

Building `fast_align` requires a modern C++ compiler and the [CMake]() build system. Additionally, the following libraries can be used to obtain better performance

 * OpenMP (included with some compilers, such as GCC)
 * libtcmalloc (part of Google's perftools)
 * libsparsehash

To install these on Ubuntu:
    
    sudo apt-get install libgoogle-perftools-dev libsparsehash-dev

To compile, do the following

    mkdir build
    cd build
    cmake ..
    make

Run `fast_align` to see a list of command line options.

`fast_align` generates *asymmetric* alignments (i.e., by treating either the left or right language in the parallel corpus as primary language being modeled, slightly different alignments will be generated). The usually recommended way to generate *source–target* (left language–right language) alignments is:

    ./fast_align -i text.fr-en -d -o -v > forward.align

The usually recommended way to generate *target–source* alignments is to just add the `-r` (“reverse”) option:

    ./fast_align -i text.fr-en -d -o -v -r > reverse.align

These can be symmetrized using the included `atools` command using a variety of standard symmetrization heuristics, for example:

    ./atools -i forward.align -j reverse.align -c grow-diag-final-and

## Output

`fast_align` produces outputs in the widely-used `i-j` “Pharaoh format,” where a pair `i-j` indicates that the <i>i</i>th word (zero-indexed) of the left language (by convention, the *source* language) is aligned to the <i>j</i>th word of the right sentence (by convention, the *target* language). For example, a good alignment of the above German–English corpus would be:

    0-0 1-1 2-4 3-2 4-3 5-5 6-6
    0-0 1-1 2-2 2-3 3-4 4-5
    0-0 1-2 2-1 3-3 4-4 5-5

## Acknowledgements

The development of this software was sponsored in part by the U.S. Army Research Laboratory and the U.S. Army Research Ofﬁce under contract/grant number W911NF-10-1-0533.

