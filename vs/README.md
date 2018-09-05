# Build Marian on Windows with GPU support


## Install prerequisites

The following SDK are required to build Marian with GPU support

   - [Cuda 9.2+](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
        - Base installer
        - Patches

   - [CuDNN 7.1+](https://developer.nvidia.com/rdp/cudnn-download)
        - Requires nVidia Developper account

   - [MKL](https://software.intel.com/en-us/mkl)

## Patch some files

### CUDA: Unsupported Visual Studio Version Error
From [nVidia forum](https://devtalk.nvidia.com/default/topic/1022648/cuda-setup-and-installation/cuda-9-unsupported-visual-studio-version-error/4)

The latest versions of Visual Studio 2017 are not officially supported by CUDA. Two fixes are proposed:
- Downgrade Visual Studio to a supported version
- Edit the file `<CUDA install path>\include\crt\host_config.h` and change the line 131:

      131     #if _MSC_VER < 1600 || _MSC_VER > 1914

  into:

      131     #if _MSC_VER < 1600 || _MSC_VER > 1915


## Configure
- Run configure.bat

## Build
- Run build.bat


## Changes from the master branch
This part gives more information on the changes done.

1. __Fix Cuda error : Unsupported Visual Studio Version Error__   
   See above for justification and fixes

2. __Fix VS compiler flags / Build in Release, with improved debug info__  
   [Github Link](https://github.com/cedrou/marian-dev/commit/1ab5f0ccb74f37b515184553c05ade523801ad9b)  
   Added VS specific compile and link flags

3. __Fix Warning: D9002: ignoring unknown option '-m64'__  
   [Github Link](https://github.com/cedrou/marian-dev/commit/5785380fa98bd61f9cae764c42116b2de39fb096)  
   This one is related to a compiler flag added while finding the package MKL that does not exists for MS compiler. 

4. __Fix marian::Backendn marian::cpu::Beckend and marian::gpu::Backend conflicts__  
   [Github Link](https://github.com/cedrou/marian-dev/commit/6370ea27d68b83c75868437bbf27bd92c9fb5628)  
   There were name conflicts between the 3 `Backend` classes that confused the compiler. I renamed the CPU and GPU as `cpuBackend` and `gpuBackend`.

5. __Fix error : identifier "CUDA_FLT_MAX" is undefined in device code__  
   [Github Link](https://github.com/cedrou/marian-dev/commit/112118fce3c8c54049913126c2685e8e7463713c)  
   `CUDA_FLT_MAX` is not seen from the device and I had to declare it as `__constant__`.

   From [StackOverflow](https://stackoverflow.com/questions/20111409/how-to-pass-structures-into-cuda-device#comment29972423_20112013):
   > Undecorated constants get compiled into both host and device code with gcc based toolchains, but not with the Microsoft compiler. 

6. __Fix fatal error C1019: unexpected #else__  
   [Github Link](https://github.com/cedrou/marian-dev/commit/5d41dff700ee1b6b5f40f7e7ceb071a306d5957c)  
   There was preprocessor instructions (`#ifdef ... #else ... #endif`) in the middle of a call of a macro function (`CUDNN_CALL`), which is not allowed with MS compiler.

7. __Fix mismatched class/struct forward declarations__  
   Microsoft's C++ name mangling makes a distinction between `class` and `struct` objects, so definitions and forward declaration must match.  
   See [this pdf](https://www.agner.org/optimize/calling_conventions.pdf), page 27, for more information.