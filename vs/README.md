# Build Marian on Windows with GPU support


## Install prerequisites

The following SDK are required to build Marian with GPU support

   - [Cuda 9.2+](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
        - Base installer
        - Patches

   - [CuDNN 7.1+](https://developer.nvidia.com/rdp/cudnn-download)
        - Requires nVidia Developper account

   - [MKL](https://software.intel.com/en-us/mkl)

### Patch for CUDA error: Unsupported Visual Studio Version Error

The latest versions of Visual Studio 2017 are not officially supported by CUDA. Two fixes are proposed:
- Downgrade Visual Studio to a supported version
- Edit the file `<CUDA install path>\include\crt\host_config.h` and change the line 131:

      131     #if _MSC_VER < 1600 || _MSC_VER > 1914

  into:

      131     #if _MSC_VER < 1600 || _MSC_VER > 1915

For more information, read this [nVidia forum](https://devtalk.nvidia.com/default/topic/1022648/cuda-setup-and-installation/cuda-9-unsupported-visual-studio-version-error/4)


## Configure
- Run configure.bat

## Build
- Run build.bat


## Changes from the master branch
This part gives more information on all changes done. Refer to [this page](https://github.com/cedrou/marian-dev/commits/build_on_win) for commits.

1. __Fix Cuda error : Unsupported Visual Studio Version Error__   
   See above for justification and fixes

2. __Fix VS compiler flags / Build in Release, with improved debug info__  
   Added VS specific compile and link flags

3. __Fix Warning: D9002: ignoring unknown option '-m64'__  
   This one is related to a compiler flag added while finding the package MKL that does not exists for MS compiler. 

4. __Fix marian::Backendn marian::cpu::Beckend and marian::gpu::Backend conflicts__  
   There were name conflicts between the 3 `Backend` classes that confused the compiler:
   
   >  template instantiation resulted in unexpected function type of "type" (the meaning of a name may have changed since the template declaration -- the type of the template is "type")
   .

   I renamed the CPU and GPU as `cpuBackend` and `gpuBackend`.

5. __Fix error : identifier "CUDA_FLT_MAX" is undefined in device code__  
   `CUDA_FLT_MAX` is not seen from the device and I had to declare it as `__constant__`.

   From [StackOverflow](https://stackoverflow.com/questions/20111409/how-to-pass-structures-into-cuda-device#comment29972423_20112013):
   > Undecorated constants get compiled into both host and device code with gcc based toolchains, but not with the Microsoft compiler. 

6. __Fix fatal error C1019: unexpected #else__  
   There was preprocessor instructions (`#ifdef ... #else ... #endif`) in the middle of a call of a macro function (`CUDNN_CALL`), which is not allowed with MS compiler.

7. __Fix mismatched class/struct forward declarations__  
   Microsoft's C++ name mangling makes a distinction between `class` and `struct` objects, so definitions and forward declaration must match.  
   See [this pdf](https://www.agner.org/optimize/calling_conventions.pdf), page 27, for more information.

8. __Fix unresolved external due to a removed #include directive__  
   An `#include` directive was specifically removed for MSVC compiler.

9. __Fix CUDA+MSVC incompatibility with Boost.Preprocessor__  
   The toolchain nvcc+msvc is not correctly handled in Boost.Preprocessor module. See [this issue](https://github.com/boostorg/preprocessor/issues/15). In the meantime, the recommended workaround is to disable Variadic Macro support in Boost.  
   I created a [PR](https://github.com/boostorg/preprocessor/pull/18) in the Boost repo on GitHub to fix this.

10. __Provide implementation for mkstemp / Fix temporary file creation__  
   The code explicitely disabled the creation of temporary files because "mkstemp not available in Windows". In fact, `_mktemp` and `_unlink` are both implemented, but thay don't work as expected. I used `_tempnam` to replace `mkstemp`, and added the flag `_O_TEMPORARY` to the parameters of `_open` to automatically delete the file when it is closed. If `unlinkEarly` is not set, I added a call to `remove` in the destructor to delete the file after its closure.  
   I also handled the case of the default value for the `base` parameter: the path `\tmp` doesnot exist on Windows, so it is replaced by the value of the `%TMP%` environment variable in `NormalizeTempPrefix`.

