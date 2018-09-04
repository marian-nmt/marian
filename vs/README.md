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