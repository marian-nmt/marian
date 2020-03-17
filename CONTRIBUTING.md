How to contribute to Marian
===========================


## Reporting a bug/asking questions

Did you find a bug or want to ask a question? Great!

* Ensure the bug was not already reported or the question was not asked on
  [the github issues](https://github.com/marian-nmt/marian-dev/issues) or
  [the Google discussion group](https://groups.google.com/forum/#!forum/marian-nmt).
* Open a new issue/question if you're unable to find yours.
* For bugs please provide as much relevant information as possible, and do not
  forget to attach training/decoding logs and your Marian command.

You are also welcome to request a new feature.
Especially if you plan to help us adding it :)


## Submitting changes

Whenever possible, please send a Github Pull Request with a clear list of what
you've done.  Feel free to update CHANGELOG.md file.  We will love you
forever if you provide unit or regression tests.

Please follow our coding convention (below) and do not forget to test your
changes by running unit tests:

    cd marian-dev/build
    cmake .. -DCOMPILE_TESTS=ON
    make test

and [regression tests](http://github.com/marian-nmt/marian-regression-tests.git):

    cd marian-dev/regression-tests
    make install
    ./run_mrt.sh


## Coding conventions

Main code style rules:

* no tabs, 2 whitespaces instead
* lines no longer than 100 characters
* no trailing whitespaces
* no space between control statements and opening brackets
* `UpperCamelCase` for class names
* `camelCaseWithTrailingUnderscore_` for class variables
* `camelCase` for variables, methods and functions
* `UPPERCASE_WITH_UNDERSCORES` for constants

You may also use [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html)
and the `.clang-format` file provided in the root directory to help you with
code formatting.

