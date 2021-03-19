# How to build Marian on Windows with GPU support


## Install prerequisites

The following SDK are required to build Marian with GPU support. At least one of them needs to be
installed. If only CUDA is installed but not MKL, a GPU-only version will be build. If only MKL is
installed and not CUDA, only the CPU version will be built. So if you are interested in only one
functionality, you can omit one of them. Install both for full functionality.

   - [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal),
     Base installer, CUDA 10.0+ is recommended, there might be issues with CUDA 9.2, see below
   - [Intel MKL](https://software.intel.com/en-us/mkl)

---
## Check dependencies : `CheckOrInstallDeps.bat`

In addition to the 2 previous prerequisites, Marian may need the following libraries that you may
already have on your system:

    - Boost (1.58-1.72), optional for marian-server (`COMPILE_SERVER=TRUE` in CMake)
    - OpenSSL, optional for marian-server

The script `CheckOrInstallDeps.bat` can be used to verify that all dependencies are found on your
system. If not, it will use the `vcpkg` library manager to download and manage your dependencies for
CMake.

If you already have a working `vcpkg` installation, this script can use it.
If vcpkg is in your `PATH` environment variable, the script will find it and use it automatically.
Otherwise, you need to edit the script and set the `VCPKG_ROOT` variable.
Please see the script for more details.

---
## Build the project

There are 3 alternatives to build the project:
1. Use Visual Studio 2017+ built-in support for CMake
2. Create a Solution file for Visual Studio
3. Use a script (MSBuild)


### 1. Use VS2017+ with built-in support for CMake

VS2017 or newer now allows to develop projects built with CMake without the need to generate VS
projects and solutions. For more information, please read [this article](https://blogs.msdn.microsoft.com/vcblog/2016/10/05/cmake-support-in-visual-studio/)
from the Visual C++ Team.

You just need to open the root folder of the git repository in VS (which contains the file
`CMakeSettings.json`):

- In an Explorer window, right-click then `Open in Visual Studio`
- In a VS2017 instance, `File > Open > Folder...`

You may need to edit the file `CMakeSettings.json` to set the environment variable for the
dependencies.

The developing experience is very similar that when using a solution file (Intellisense, build
project with `F7`, debug, set breakpoints and watch variables, ...), except that the project
configuration is done in 3 different files:

   - `CMakeList.txt`: this is the CMake source file from the original project.
     It is used to configure the build targets, add/remove files to compile and configure the
     compiler flags.

   - `CMakeSettings.json`: this file is required to enable CMake integration in VS2017.
     Use this file to configure the environment variables and the parameters passed to CMake to
     generate the project.

   - `.vs\launch.vs.json`: this is an optional user specific file and it is not commited in the Git
     repo. Use this file to configure the debugging targets. For example:

         {
             "version": "0.2.1",
             "defaults": {},
             "configurations": [
                 {
                 "type": "default",
                 "name": "Training Basics",
                 "project": "CMakeLists.txt",
                 "projectTarget": "marian.exe",
                 "currentDir": "D:\\Perso\\github\\marian\\marian-examples\\training-basics",
                 "args": [
                     "--devices 0",
                     "--type amun",
                     "--model model/model.npz",
                     "--train-sets data/corpus.bpe.ro data/corpus.bpe.en",
                     "--vocabs model/vocab.ro.yml model/vocab.en.yml",
                     "--dim-vocabs 66000 50000",
                     "--mini-batch-fit",
                     "-w 3000",
                     "--layer-normalization",
                     "--dropout-rnn 0.2",
                     "--dropout-src 0.1",
                     "--dropout-trg 0.1",
                     "--early-stopping 5",
                     "--valid-freq 100",
                     "--save-freq 10000",
                     "--disp-freq 100",
                     "--valid-metrics cross-entropy translation",
                     "--valid-sets data/newsdev2016.bpe.ro data/newsdev2016.bpe.en",
                     "--valid-script-path .\\scripts\\validate.bat",
                     "--log model/train.log",
                     "--valid-log model/valid.log",
                     "--overwrite",
                     "--keep-best",
                     "--seed 1111",
                     "--exponential-smoothing",
                     "--normalize 1",
                     "--beam-size 12",
                     "--quiet-translation"
                 ]
                 }
             ]
         }


### 2. Create solution and projects files for Visual Studio : `CreateVSProjects.bat`

If you have a previous version of Visual Studio, you will need to use CMake to generate the projects
files.

The provided script `CreateVSProjects.bat` runs the dependency checks then invokes CMake with the
right parameters to create the solutions for Visual Studio.


### 3. Use MSBuild : `BuildRelease.bat`

The last alternative is to use the script `BuildRelease.bat` that will:
- Check the dependencies
- Create the VS project files
- Invoke MSBuild on these projects to build the targets in Release.

---
## Known issues

1. __Patch for CUDA 9.2 error: Unsupported Visual Studio Version Error__

    When using CUDA 9.2, the latest versions of Visual Studio 2017 are not officially supported by
    CUDA. Two fixes are proposed:
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
