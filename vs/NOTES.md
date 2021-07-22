# How to build Marian on Windows with GPU support

This is interesting mostly for developers. Warning: it has been extracted from
an old `vs/README.md` and some information might be outdated.

---
## Known issues

1. __Patch for CUDA 9.2 error: Unsupported Visual Studio Version Error__

    When using CUDA 9.2, the latest versions of Visual Studio 2017 are not
    officially supported by CUDA. Two fixes are proposed:
    - Downgrade Visual Studio to a supported version
    - Edit the file `<CUDA install path>\include\crt\host_config.h` and change the line 131:

            131     #if _MSC_VER < 1600 || _MSC_VER > 1914

        into:

            131     #if _MSC_VER < 1600 || _MSC_VER > 1915

    For more information, read this [nVidia forum](https://devtalk.nvidia.com/default/topic/1022648/cuda-setup-and-installation/cuda-9-unsupported-visual-studio-version-error/4)

2. __It does not compile with Boost 1.73 or newer__

    It may happen that SimpleWebSocketServer, a 3rd party library that Marian uses for
    marian-server, does not support the version of Boost available in vcpkg. In such case install a
    supported version of Boost; if you use vcpkg, an option is to checkout to #5970385, which has
    Boost 1.72.

    Note that Boost is required only if you compile with marian-server, for compilation using CMake,
    it is if you set `COMPILE_SERVER` to `TRUE` in CMakeSettings.json.

---
## Changes from the master branch
This part gives more information on all changes done in this PR. Refer to [this page](https://github.com/cedrou/marian-dev/commits/build_on_win) for commits.

1. __Fix Cuda error : Unsupported Visual Studio Version Error__
   See above for justification and fixes

2. __Fix VS compiler flags / Build in Release, with improved debug info__
   Added VS specific compile and link flags

3. __Fix Warning: D9002: ignoring unknown option '-m64'__
   This one is related to a compiler flag added while finding the package MKL that does not exists for MS compiler.

4. __Fix marian::Backend, marian::cpu::Backend and marian::gpu::Backend conflicts__
   There were name conflicts between the 3 `Backend` classes that confused the compiler:

   >  template instantiation resulted in unexpected function type of "void(Ptr\<marian::gpu::Backend\> backend, [...])" (the meaning of a name may have changed since the template declaration -- the type of the template is "void(Ptr\<marian::Backend\> backend, [...]").

   To solve this, I changed the declaration of 3 methods to specify the full name with namespace (`marian::Backend`, instead of `Backend`).

5. __Fix error : identifier "CUDA_FLT_MAX" is undefined in device code__
   `CUDA_FLT_MAX` is not seen by CUDA from the device code and I had to declare it as `__constant__`.

   From [StackOverflow](https://stackoverflow.com/questions/20111409/how-to-pass-structures-into-cuda-device#comment29972423_20112013):
   > Undecorated constants get compiled into both host and device code with gcc based toolchains, but not with the Microsoft compiler.

6. __Fix fatal error C1019: unexpected #else__
   There was preprocessor instructions (`#ifdef ... #else ... #endif`) in the middle of a call of a macro function (`CUDNN_CALL`), which is not allowed with MS compiler.

7. __Fix mismatched class/struct forward declarations__
   Microsoft's C++ name mangling makes a distinction between `class` and `struct` objects, so definitions and forward declaration must match.
   See [this pdf](https://www.agner.org/optimize/calling_conventions.pdf), page 27, for more information.

   _Note_: This fix was invalidated by commit # from @frankseide

8. __Fix unresolved external due to a removed #include directive__
   There was an include directive removed from MSVC compilation, but this prevented the build of the project.
   I'm not sure why this was removed; the comment is:

        #ifndef _WIN32  // TODO: remove this once I updated the Linux-side makefile

9. __Fix CUDA+MSVC incompatibility with Boost.Preprocessor__
   The toolchain nvcc+msvc is not correctly handled in Boost.Preprocessor module. See [this issue](https://github.com/boostorg/preprocessor/issues/15). In the meantime, the recommended workaround is to disable Variadic Macro support in Boost.
   I created a [PR](https://github.com/boostorg/preprocessor/pull/18) in the Boost repo on GitHub to fix this.

   _Note_: The library sources have been fixed, but this fix is still needed until the next release of Boost.Preprocessor

10. __Provide implementation for mkstemp / Fix temporary file creation__
   The code explicitely disabled the creation of temporary files because "mkstemp not available in Windows". In fact, `mktemp` and `unlink` are both implemented, but they don't work as expected. I used `tempnam` to replace `mkstemp`, and added the flag `_O_TEMPORARY` to the parameters of `open` to automatically delete the file when it is closed. If `unlinkEarly` is not set, I added a call to `remove` in the destructor to delete the file after its closure.
   I also handled the case of the default value for the `base` parameter: the path `\tmp` doesnot exist on Windows, so it is replaced by the value of the `%TMP%` environment variable in `NormalizeTempPrefix`.

11. __Revert commit #2f8b093 + Fix copy/paste error while fixing #301 + restrict fix to MSVC compiler.__
   cf [Issue #301](https://github.com/marian-nmt/marian-dev/issues/301)   -->

