
# Marian

[![CUDABuild Status](http://vali.inf.ed.ac.uk/jenkins/buildStatus/icon?job=amunmt_compilation_cuda)](http://vali.inf.ed.ac.uk/jenkins/job/amunmt_compilation_cuda/)
[![CPU Build Status](http://vali.inf.ed.ac.uk/jenkins/buildStatus/icon?job=amunmt_compilation_cpu)](http://vali.inf.ed.ac.uk/jenkins/job/amunmt_compilation_cpu/)
[![Twitter](https://img.shields.io/twitter/follow/marian_nmt.svg?style=social&label=Follow)](https://twitter.com/intent/follow?screen_name=marian_nmt)


 <p>
  <b>Marian</b> (formerly known as AmuNMT) is an efficient Neural Machine Translation framework written
  in pure C++ with minimal dependencies. It has mainly been developed at the
  Adam Mickiewicz University in Poznań (AMU) and at the University of Edinburgh.
  </p>

  <p>
  It is currently being deployed in
  multiple European projects and is the main translation and training engine
  behind the neural MT launch at the
  <a href="http://www.wipo.int/pressroom/en/articles/2016/article_0014.html">World Intellectual Property Organization</a>.

  </p>

  <p>
  Main features:
  <ul>
    <li> Fast multi-gpu training and translation </li>
    <li> Compatible with Nematus and DL4MT </li>
    <li> Efficient pure C++ implementation </li>
    <li> Permissive open source license (MIT) </li>
    <li> <a href="https://marian-nmt.github.io/features/"> more details... </a> </li>
  </ul>
  </p>

If you use this, please cite:

Marcin Junczys-Dowmunt , Roman Grundkiewicz, Tomasz Dwojak, Hieu Hoang, Kenneth Heafield, Tom Neckermann, Frank Seide, Ulrich Germann, Alham Fikri Aji, Nikolay Bogoychev, André F. T. Martins, Alexandra Birch (2018). Marian: Fast Neural Machine Translation in C++ (https://arxiv.org/abs/1804.00344)

    @article{junczys2018marian,
      title={Marian: Fast Neural Machine Translation in C++},
      author={Marcin Junczys-Dowmunt and Roman Grundkiewicz and Tomasz Dwojak and Hieu Hoang and Kenneth Heafield and Tom Neckermann and Frank Seide and Ulrich Germann and Alham Fikri Aji and Nikolay Bogoychev and André F. T. Martins and Alexandra Birch},
      journal={arXiv preprint arXiv:1804.00344},
      url={https://arxiv.org/abs/1804.00344}
      year={2018}
    }

## Website:

More information on https://marian-nmt.github.io

## Recommended software

### GPU version

**Ubuntu 16.04 LTS (tested and recommended).** For Ubuntu 16.04 the standard
packages should work. On newer versions of Ubuntu, e.g. 16.10, there may be
problems due to incompatibilities of the default g++ compiler and CUDA.

 * CMake 3.5.1 (default)
 * GCC/G++ 5.4 (default)
 * Boost 1.58 (default)
 * CUDA 8.0

**Ubuntu 14.04 LTS.** A newer CMake version than the default version is
required and can be installed from source.

 * CMake 3.5.1 (due to CUDA related bugs in earlier versions)
 * GCC/G++ 4.9
 * Boost 1.54
 * CUDA 8.0

### CPU version

The CPU-only version will automatically be compiled if CUDA cannot be detected by CMake.
Only the translator will be compiled, the training framework is strictily GPU-based.

Tested on different machines and distributions:

 * CMake 3.5.1
 * The CPU version should be a lot more forgiving concerning GCC/G++ or Boost versions.

#### macOS

To be able to make the CPU version on macOS, first install [brew](https://brew.sh/) and then run:

    brew install cmake boost

    # Python 2 default
    brew install boost-python

    # Python 3
    brew install boost-python --with-python3

Then, proceed to the next section.

## Download and Compilation

Clone a fresh copy from github:

    git clone https://github.com/marian-nmt/marian.git

The project is a standard CMake out-of-source build:

    cd marian
    mkdir build && cd build
    cmake ..
    make -j

If run for the first time, this will also download Marian -- the training
framework for Marian.

Other cmake options:

-  Build the CPU-only version of `amun` (training is GPU-only)

       cmake .. -DCUDA=off

-  Adding debugging symbols (for use with gdb, etc)

       cmake .. -DCMAKE_BUILD_TYPE=Debug

- Specifying Python version to compile against

       # Linux
       cmake .. -DPYTHON_VERSION=2.7
       cmake .. -DPYTHON_VERSION=3.5
       cmake .. -DPYTHON_VERSION=3.6

       # macOS
       cmake .. -DPYTHON_VERSION=2
       cmake .. -DPYTHON_VERSION=3


### Compile Python bindings

In order to compile the Python library, after running _make_ as in the previous section, do:

    make python

This will generate a _libamunmt.dylib_ or _libamunmt.so_ in your `build/src/` directory, which can be imported from Python.

## Running Marian

### Training

Assuming `corpus.en` and `corpus.ro` are
corresponding and preprocessed files of a English-Romanian parallel corpus, the
following command will create a Nematus-compatible neural machine translation model.

    ./marian/build/marian \
      --train-sets corpus.en corpus.ro \
      --vocabs vocab.en vocab.ro \
      --model model.npz

See the [documentation](https://marian-nmt.github.io/docs/#marian) for a full list
of command line options or the
[examples](https://marian-nmt.github.io/examples/training) for a full example of
how to train a WMT-grade model.

### Translating

If a trained model is available, run:

    ./marian/build/amun -m model.npz -s vocab.en -t vocab.ro <<< "This is a test ."

See the [documentation](https://marian-nmt.github.io/docs/#amun) for a full list of
command line options or the
[examples](https://marian-nmt.github.io/examples/translating) for a full example of
how to use Edinburgh's WMT models for translation.

## Example usage

* **[Translating with Amun](https://marian-nmt.github.io/examples/translating/)**:
The files and scripts described in this section can be found in
`amunmt/examples/translate`. They demonstrate how to translate with Amun using
Edinburgh's German-English WMT2016 single model and ensemble.
* **[Training with Marian](https://marian-nmt.github.io/examples/training/)**: The files
and scripts described in this section can be found in
`marian/examples/training`. They have been adapted from the
Romanian-English sample from <https://github.com/rsennrich/wmt16-scripts>.
We also add the back-translated data from <http://data.statmt.org/rsennrich/wmt16_backtranslations/>
as desribed in [Edinburgh's WMT16 paper](http://www.aclweb.org/anthology/W16-2323).
The resulting system should be competitive or even slightly better than
reported in that paper.
* **[Winning system of the WMT 2016 APE shared task](https://marian-nmt.github.io/examples/postedit/)**:
This page provides data and model files for our shared task winning APE system
described in [Log-linear Combinations of Monolingual and Bilingual Neural
Machine Translation Models for Automatic
Post-Editing](http://www.aclweb.org/anthology/W16-2378).

## Acknowledgements

The development of Marian received funding from the European Union's
_Horizon 2020 Research and Innovation Programme_ under grant agreements
688139 ([SUMMA](http://www.summa-project.eu); 2016-2019),
645487 ([Modern MT](http://www.modernmt.eu); 2015-2017),
644333 ([TraMOOC](http://tramooc.eu/); 2015-2017),
644402 ([HiML](http://www.himl.eu/); 2015-2017),
the Amazon Academic Research Awards program,
the World Intellectual Property Organization,
and is based upon work supported in part by the Office of the Director of
National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
(IARPA), via contract #FA8650-17-C-9117.

This software contains source code provided by NVIDIA Corporation.

