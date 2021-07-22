# Building Marian on Windows


## Install prerequisites

At least one of the following SDK is required to build Marian on Windows:

   - [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal),
     Base installer, CUDA 10.0+ is recommended, there might be issues with CUDA
     9.2, see below
   - [Intel MKL](https://software.intel.com/en-us/mkl)

CUDA is required for running Marian on GPU, and Intel MKL for CPU. If only one
of them is installed, a GPU-only or CPU-only version can be built.

---
## Check dependencies : `CheckOrInstallDeps.bat`

The script `CheckOrInstallDeps.bat` can be used to verify that all dependencies
are found on your system. If not, it will use the `vcpkg` library manager to
download and manage your dependencies for CMake, including the following
optional libraries needed only if you want to compile Marian server:

    - Boost (1.58-1.72), optional for marian-server (`COMPILE_SERVER=TRUE` in
      `CMakeSettings.json`)
    - OpenSSL, optional for marian-server

If you already have a working `vcpkg` installation, this script can use it. If
vcpkg is in your `PATH` environment variable, the script will find it and use
it automatically. Otherwise, you need to edit the script and set the
`VCPKG_ROOT` variable. Please see the script for more details.

---
## Build the project

There are 3 alternatives to build the project:
1. Use Visual Studio 2017+ built-in support for CMake
2. Create a Solution file for Visual Studio
3. Use a script (MSBuild)


### 1. Use VS2017+ with built-in support for CMake

VS2017 or newer now allows to develop projects built with CMake without the
need to generate VS projects and solutions. For more information, please read
[this article](https://blogs.msdn.microsoft.com/vcblog/2016/10/05/cmake-support-in-visual-studio/)
from the Visual C++ Team.

1. Open the root folder of the git repository in VS (which contains the file
   `CMakeSettings.json`) using `Open local folder` on the welcome page or `File
   > Open > Folder...` in a VS instance.
2. Edit the file `CMakeSettings.json` to set the environment variable for the
   dependencies. Set `COMPILE_CUDA` or `COMPILE_CPU` to `FALSE` if you wish to
   compile a CPU-only or a GPU-only version respectively.
3. VS2017 should automatically detect `CMakeSettings.json` and generate CMake
   Cache.
4. Build the project with `F7`. If build is successful, the executables will be
   in the `build` folder.


#### Development

The developing experience is very similar that when using a solution file
(Intellisense, build project with `F7`, debug, set breakpoints and watch
variables, ...), except that the project configuration is done in 3 different
files:

- `CMakeList.txt`: this is the CMake source file from the original project.
  It is used to configure the build targets, add/remove files to compile and configure the
  compiler flags.

- `CMakeSettings.json`: this file is required to enable CMake integration in VS2017.
  Use this file to configure the environment variables and the parameters passed to CMake to
  generate the project.

- `.vs\launch.vs.json`: this is an optional user specific file and it is not commited in the Git
  repo. Use this file to configure the debugging targets.


### 2. Create solution and projects files for Visual Studio : `CreateVSProjects.bat`

If you have a previous version of Visual Studio, you will need to use CMake to
generate the projects
files.

The provided script `CreateVSProjects.bat` runs the dependency checks then
invokes CMake with the right parameters to create the solutions for Visual
Studio.

Warning: the Visual Studio Solution file included in the `vs/` folder might not
work out of the box with your environment and require customization.


### 3. Use MSBuild : `BuildRelease.bat`

The last alternative is to use the script `BuildRelease.bat` that will:
- Check the dependencies.
- Create the VS project files.
- Invoke MSBuild on these projects to build the targets in Release.
