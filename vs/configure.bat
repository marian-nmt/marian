@echo off
setlocal

set ROOT=%~dp0
set MARIAN_ROOT=%ROOT%..

:: If you already have vcpkg (a C++ library manager for Windows), and it is not in your PATH
:: set the following variable to the directory that contains the vcpkg.exe
set VCPKG_ROOT=


:: Clone or update vcpkg (used to manage the dependencies)
:: -------------------------------------------------------
if "%VCPKG_ROOT%" == "" for /f "delims=" %%p in ('where vcpkg 2^>nul') do set VCPKG_ROOT=%%~dpp

if "%VCPKG_ROOT%" == "" set VCPKG_ROOT=%ROOT%deps\vcpkg

if not exist %VCPKG_ROOT% (
    echo --- Cloning vcpkg...
    git clone https://github.com/Microsoft/vcpkg.git %VCPKG_ROOT%

    set BOOTSTRAP_VCPKG=1
) else (
    echo --- Updating vcpkg...
    pushd %VCPKG_ROOT%
    git pull >nul
    for /f "delims=" %%p in ('git pull') do (
        if not "%%p" == "Already up to date." set BOOTSTRAP_VCPKG=1
    )
    popd
)

if "%BOOTSTRAP_VCPKG%" == "1" (
    pushd %VCPKG_ROOT%
    call bootstrap-vcpkg.bat
    popd
)


set VCPKG_DEFAULT_TRIPLET=x64-windows-static
set VCPKG_INSTALL=%VCPKG_ROOT%\installed\%VCPKG_DEFAULT_TRIPLET%
set VCPKG=%VCPKG_ROOT%\vcpkg

echo.


:: Check dependencies and configure CMake
:: -------------------------------------------------------

echo --- Checking dependencies...

set CMAKE_OPT=


:: Use vcpkg toolchain
set CMAKE_OPT=%CMAKE_OPT% -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
set CMAKE_OPT=%CMAKE_OPT% -DVCPKG_TARGET_TRIPLET=%VCPKG_DEFAULT_TRIPLET%


::
:: Check prerequisites
::

:: -------------------------
echo ... CUDA
if not exist "%CUDA_PATH%" (
    echo The CUDA_PATH environment variable is not defined: please make sure CUDA 8.0+ is installed.
    goto :eof
)
echo Found Cuda SDK in %CUDA_PATH%


:: -------------------------
echo ... CUDNN
if not exist "%CUDA_PATH%\lib\x64" (
    echo CuDNN not found in your CUDA installation
    goto :eof
)
echo Found CuDNN in "%CUDA_PATH%\lib\x64"

set CMAKE_OPT=%CMAKE_OPT% -D CUDNN_LIBRARY:PATH="%CUDA_PATH%\lib\x64\cudnn.lib"
set CMAKE_OPT=%CMAKE_OPT% -D CUDNN_INCLUDE:PATH="%CUDA_PATH%\include"


:: -------------------------
echo ... Intel MKL
set MKLROOT="C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl"
if not exist %MKLROOT% (
    echo Please modify the script file to set MKLROOT to the installation path of the Intel MKL library.
    goto :eof
)
echo Found Intel MKL in %MKLROOT%

set CMAKE_OPT=%CMAKE_OPT% -DMKL_ROOT:PATH=%MKLROOT%
set CMAKE_OPT=%CMAKE_OPT% -DMKL_INCLUDE_DIRS:PATH=%MKLROOT%\include
set CMAKE_OPT=%CMAKE_OPT% -DMKL_CORE_LIBRARY:FILEPATH=%MKLROOT%\lib\intel64\mkl_core.lib
set CMAKE_OPT=%CMAKE_OPT% -DMKL_INTERFACE_LIBRARY:FILEPATH=%MKLROOT%\lib\intel64\mkl_intel_ilp64.lib
set CMAKE_OPT=%CMAKE_OPT% -DMKL_SEQUENTIAL_LAYER_LIBRARY:FILEPATH=%MKLROOT%\lib\intel64\mkl_sequential.lib


:: -------------------------
echo ... Boost (1.58+)
%VCPKG% install boost-chrono boost-filesystem boost-iostreams boost-program-options boost-regex boost-system boost-thread boost-timer boost-asio

:: -------------------------
echo ... zlib
%VCPKG% install zlib

:: -------------------------
echo ... OpenSSL
%VCPKG% install openssl
set CMAKE_OPT=%CMAKE_OPT% -DOPENSSL_USE_STATIC_LIBS:BOOL=TRUE
set CMAKE_OPT=%CMAKE_OPT% -DOPENSSL_MSVC_STATIC_RT:BOOL=TRUE

echo.

echo --- Configuring CMake...

set CMAKE_OPT=%CMAKE_OPT% -DBUILD_STATIC:BOOL=TRUE 

:: -----  Target Visual Studio 2017 64bits -----
set CMAKE_OPT=%CMAKE_OPT% -G"Visual Studio 15 2017 Win64" 

:: Policy CMP0074: find_package uses <PackageName>_ROOT variables.
set CMAKE_OPT=%CMAKE_OPT% -D CMAKE_POLICY_DEFAULT_CMP0074=NEW

:: -----  Disable some tool build -----
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_TRAIN:BOOL=TRUE 
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_DECODER:BOOL=TRUE 
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_SERVER:BOOL=TRUE 
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_SCORER:BOOL=FALSE 
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_PYTHON:BOOL=FALSE 
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_VOCAB:BOOL=TRUE 
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_EXAMPLES:BOOL=FALSE 
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_TESTS:BOOL=FALSE 

set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_CPU:BOOL=TRUE 
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_CUDA:BOOL=TRUE 

set CMAKE_OPT=%CMAKE_OPT% -D USE_CUDNN:BOOL=TRUE 
set CMAKE_OPT=%CMAKE_OPT% -D USE_MPI:BOOL=FALSE

echo.
echo.
echo --------------------------------------------------
echo CMAKE configuration flags:
echo %CMAKE_OPT%
echo --------------------------------------------------
echo.
echo.

set BUILD_ROOT=%ROOT%build-vs
if not exist %BUILD_ROOT% mkdir %BUILD_ROOT%
pushd %BUILD_ROOT%


echo.
echo --- Creating Visual Studio projects...
cmake %CMAKE_OPT% %MARIAN_ROOT%

::echo.
::echo --- Building projects...
::cmake --build .

popd
