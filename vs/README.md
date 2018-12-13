# How to build Marian on Windows with GPU support


## Install prerequisites

The following SDK are required to build Marian with GPU support. At least one of them needs to be installed. If only CUDA is installed but not MKL,
a GPU-only version will be build. If only MKL is installed and not CUDA, only the CPU version will be built. So if you are interested in only one
functionality, you can ommit one of them. Install both for full functionality. 

   - [Cuda 10](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
        - Base installer

   - [MKL](https://software.intel.com/en-us/mkl)


__Note: Patch for CUDA 9.2 error: Unsupported Visual Studio Version Error__

This seems to work fine with CUDA 10.0.

When using CUDA 9.2, the latest versions of Visual Studio 2017 are not officially supported by CUDA. Two fixes are proposed:

   - Downgrade Visual Studio to a supported version

   - Edit the file `<CUDA install path>\include\crt\host_config.h` and change the line 131:

         131     #if _MSC_VER < 1600 || _MSC_VER > 1914

     into:

         131     #if _MSC_VER < 1600 || _MSC_VER > 1915

For more information, read this [nVidia forum](https://devtalk.nvidia.com/default/topic/1022648/cuda-setup-and-installation/cuda-9-unsupported-visual-studio-version-error/4)

---
## Check dependencies : `CheckDeps.bat`

In addition to the 2 previous prerequisites, Marian needs 2 libraries that you may already have on your system:

    - Boost (1.58+)
    - OpenSSL (optional for server)

The script `CheckDeps.bat` can be used to verify that all dependencies are found on your system. If not, it will use the `vcpkg` library manager to download and manage your dependencies for CMake.

If you already have a working `vcpkg` installation, this script can use it:
- If vcpkg is in your `PATH` environment variable, the script will find it and use it automatically.
- Otherwise, you need to edit the script and set the `VCPKG_ROOT` variable to the directory that contains the vcpkg.exe


If you prefer to manage yourself the dependencies, you can edit the script file to set the following variables to the respective installation paths. These variable can also be already set in your environment.
- `BOOST_INCLUDE_PATH` and `BOOST_LIB_PATH`
- `OPENSSL_PATH`

---
## Build the project

There are 3 alternatives to build the project:
1. Use Visual Studio 2017 built-in support for CMake
2. Create a Solution file for Visual Studio
3. Use a script (MSBuild)

### 1. Use VS2017 with built-in support for CMake

VS2017 now allows to develop projects built with CMake without the need to generate VS projects and solutions. For more information, please read [this article](https://blogs.msdn.microsoft.com/vcblog/2016/10/05/cmake-support-in-visual-studio/) from the Visual C++ Team.

You just need to open the root folder of the git repository in VS (which contains the file `CMakeSettings.json`):
- In an Explorer window, right-click then `Open in Visual Studio`
- In a VS2017 instance, `File > Open > Folder...`

You may need to edit the file `CMakeSettings.json` to set the environment variable for the dependencies.

The developing experience is very similar than when using a solution file (Intellisense, build project with `F7`, debug, set breakpoints and watch variables, ...), except that the project configuration is done in 3 different files:

   - `CMakeList.txt`: this is the CMake source file from the original project.  
     It is used to configure the build targets, add/remove files to compile and configure the compiler flags.

   - `CMakeSettings.json`: this file is required to enable CMake integration in VS2017.  
     Use this file to configure the environment variables and the parameters passed to CMake to generate the project.
   
   - `.vs\launch.vs.json`: this is a user specific file and it is not commited in the Git repo  
     Use this file to configure the debugging targets.  
     For example:

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
                     "--normalize=1",
                     "--beam-size=12",
                     "--quiet-translation"
                 ]
                 }
             ]
         }



### 2. Create solution and projects files for Visual Studio : `CreateVSProjects.bat`

If you have a previous version of Visual Studio, you will need to use CMake to generate the projects files.

The provided script `CreateVSProjects.bat` runs the dependency checks then invokes CMake with the right parameters to create the solutions for Visual Studio.


### 3. Use MSBuild : `BuildRelease.bat`

The last alternative is to use the script `BuildRelease.bat` that will:
- Check the dependencies
- Create the VS project files
- Invoke MSBuild on these projects to build the targets in Release.

<!-- 
This is interesting for developers, hiding away from users.

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
   
   