Contributing to Marian
======================

## Code style

Main code style rules:

* no tabs, 2 whitespaces instead
* lines no longer than 80 characters
* no trailing whitespaces
* no space between control statements and opening brackets

Alternatively, use the provided `.clang-format` file for
[ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) to format new code
fragments, e.g.

  clang-format-3.8 <path_to_file>

*Notice*: If you want to merge your changes from the repository cloned before
autoformatting on all files in the _master_ repository, you may want to use the
`autoformat.sh` script on your files before the merge.
