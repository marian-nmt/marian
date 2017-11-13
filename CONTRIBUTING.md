How to contribute to Marian
===========================


## Reporting a bug/asking questions

Did you find a bug or want to ask a question? Great!

* Ensure the bug was not already reported or the question was not asked
* For bugs check the [github
  issues](https://github.com/marian-nmt/marian-dev/issues)
* For answers to your question search posts on the [Google discussion
  group](https://groups.google.com/forum/#!forum/marian-nmt)
* Open a new issue/question if you're unable to find yours
* For bugs please provide as much relevant information as possible, and do not
  forget to paste a log

You are also welcome to request a new feature.
Especially if you plan to help us adding it :)


## Submitting changes

Whenever possible, please send a Github Pull Request with a clear list of what
you've done.  Feel free to also update CHANGELOG.md file.  We will love you
forever if you provide unit or regression tests.

Please follow our coding convention (below) and make sure all of your commits
are atomic (learn more about _git squash_ to merge multiple commits and _git
rebase -i_ to split a single huge commit into smaller pieces).

Ideally test your changes by running [Marian regression
tests](http://github.com/marian-nmt/marian-regression-tests.git) locally:

    git clone http://github.com/marian-nmt/marian-regression-tests.git
    cd marian-regression-tests.git
    make BRANCH=<your_branch_name> install
    ./run_mrt.sh


## Coding conventions

Main code style rules:

* no tabs, 2 whitespaces instead
* lines no longer than 80 characters
* no trailing whitespaces
* no space between control statements and opening brackets
* `UpperCamelCase` for class names
* `camelCaseWithTrailingUnderscore_` for class variables
* `camelCase` for variables, methods and functions
* `UPPERCASE_WITH_UNDERSCORES` for constants

Ideally, use the provided `.clang-format` file (in the root directory) for
[ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) to format your code, e.g.

    clang-format-3.8 <path_to_file>

