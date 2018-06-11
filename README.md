Marian
======

[![Join the chat at https://gitter.im/marian-nmt](https://badges.gitter.im/amunmt/marian.svg)](https://gitter.im/marian-nmt?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](http://vali.inf.ed.ac.uk/jenkins/buildStatus/icon?job=marian-dev)](http://vali.inf.ed.ac.uk/jenkins/job/marian-dev/)
[![Tests Status](http://vali.inf.ed.ac.uk/jenkins/buildStatus/icon?job=marian-regression-tests)](http://vali.inf.ed.ac.uk/jenkins/job/marian-regression-tests/)
[![Twitter](https://img.shields.io/twitter/follow/marian_nmt.svg?style=social&label=Follow)](https://twitter.com/intent/follow?screen_name=marian_nmt)

**Marian** is a C++ GPU-specific parallel automatic differentiation library
with operator overloading. It is the training framework used in the Marian
toolkit. This repository is the development repo of
https://github.com/marian-nmt/marian, use it at your own risk.

https://github.com/marian-nmt/marian is updated with stable versions of this
repository.

Named in honour of Marian Rejewski, a Polish mathematician and cryptologist.

## Compilation

```
cd marian-dev
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=relwithdebinfo
make -j
```

## Website

More information on https://marian-nmt.github.io

## Mailing lists

* General google group: https://groups.google.com/forum/#!forum/marian-nmt (for users and developers)
* Google group for commit messages: https://groups.google.com/forum/#!forum/marian-nmt-commits (for developers)

## Contributions

See [CONTRIBUTING.md](https://github.com/marian-nmt/marian-dev/blob/master/CONTRIBUTING.md)

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

