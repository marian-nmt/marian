@echo off
setlocal

set ROOT=%~dp0
set MARIAN_ROOT=%ROOT%..


set NEED_VCPKG=
if "%BOOST_INCLUDE_PATH%" == "" set NEED_VCPKG=1
if "%ZLIB_PATH%" == "" set NEED_VCPKG=1
if "%OPENSSL_PATH%" == "" set NEED_VCPKG=1

if "%NEED_VCPKG%"=="" goto :checkDeps

:: If you already have vcpkg (a C++ library manager for Windows), and it is not in your PATH
:: Please set the VCPKG_ROOT variable to the directory that contains the vcpkg.exe


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



:checkDeps

:: Check dependencies and configure CMake
:: -------------------------------------------------------

echo.
echo --- Checking dependencies...

set CMAKE_OPT=


:: -------------------------
:: The CUDA_PATH env variable is normally set by the CUDA SDK installer
::
echo.
echo ... CUDA
if "%CUDA_PATH%"=="" (
    echo The CUDA_PATH environment variable is not defined: please make sure CUDA 8.0+ is installed.
    goto :eof
)
if not exist "%CUDA_PATH%" (
    echo CUDA_PATH is set to an invalid path:
    echo %CUDA_PATH%
    echo Please make sure CUDA 8.0+ is properly installed.
    goto :eof
)

echo Found Cuda SDK in %CUDA_PATH%


:: -------------------------
::
echo.
echo ... CUDNN
if not exist "%CUDA_PATH%\lib\x64\cudnn.lib" (
    echo CuDNN not found in your CUDA installation
) else (
    echo Found CuDNN library in %CUDA_PATH%\lib\x64

    set CMAKE_OPT=%CMAKE_OPT% -D CUDNN_LIBRARY:PATH="%CUDA_PATH%\lib\x64\cudnn.lib"
)


:: -------------------------
:: If MKL_PATH is not set, we use the standard default installation dir
::
echo.
echo ... Intel MKL
if "%MKL_PATH%" == "" ( 
    set "MKL_PATH=C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl"
)
if not exist "%MKL_PATH%" (
    echo MKL_PATH is set to an invalid path:
    echo "%MKL_PATH%"
    echo Please set MKL_PATH to the installation path of the Intel MKL library.
    goto :eof
)
echo Found Intel MKL library in %MKL_PATH%

set CMAKE_OPT=%CMAKE_OPT% -DMKL_ROOT:PATH="%MKL_PATH%"
set CMAKE_OPT=%CMAKE_OPT% -DMKL_INCLUDE_DIRS:PATH="%MKL_PATH%\include"
set CMAKE_OPT=%CMAKE_OPT% -DMKL_CORE_LIBRARY:FILEPATH="%MKL_PATH%\lib\intel64\mkl_core.lib"
set CMAKE_OPT=%CMAKE_OPT% -DMKL_INTERFACE_LIBRARY:FILEPATH="%MKL_PATH%\lib\intel64\mkl_intel_ilp64.lib"
set CMAKE_OPT=%CMAKE_OPT% -DMKL_SEQUENTIAL_LAYER_LIBRARY:FILEPATH="%MKL_PATH%\lib\intel64\mkl_sequential.lib"


:: -------------------------
:: BOOST_INCLUDE_PATH and BOOST_LIB_PATH can be both set to an existing Boost installation.
:: If not, we use vcpkg to install the required Boost packages
::
echo.
echo ... Boost (1.58+)
if "%BOOST_INCLUDE_PATH%" == "" (
    "%VCPKG%" install boost-chrono boost-filesystem boost-iostreams boost-program-options boost-regex boost-system boost-thread boost-timer boost-asio
    set BOOST_INCLUDE_PATH="%VCPKG_INSTALL%\include"
    set BOOST_LIB_PATH="%VCPKG_INSTALL%\lib"
)
if not exist "%BOOST_INCLUDE_PATH%" (
    echo BOOST_INCLUDE_PATH is set to an invalid path:
    echo "%BOOST_INCLUDE_PATH%"
    echo Please set BOOST_INCLUDE_PATH and BOOST_LIB_PATH to the installation path of the Boost library.
    goto :eof
)
if not exist "%BOOST_LIB_PATH%" (
    echo BOOST_LIB_PATH is set to an invalid path:
    echo "%BOOST_LIB_PATH%"
    echo Please set BOOST_INCLUDE_PATH and BOOST_LIB_PATH to the installation path of the Boost library.
    goto :eof
)
echo Found Boost headers in "%BOOST_INCLUDE_PATH%"

set CMAKE_OPT=%CMAKE_OPT% -D BOOST_INCLUDEDIR:PATH="%BOOST_INCLUDE_PATH%"
set CMAKE_OPT=%CMAKE_OPT% -D BOOST_LIBRARYDIR:PATH="%BOOST_LIB_PATH%"


:: -------------------------
:: ZLIB_PATH can be set to an existing zlib installation.
:: If not, we use vcpkg to install the library
::
echo.
echo ... zlib
if "%ZLIB_PATH%"=="" (
    %VCPKG% install zlib
    set ZLIB_PATH=%VCPKG_INSTALL%
)

set CMAKE_OPT=%CMAKE_OPT% -D ZLIB_LIBRARY:PATH=%ZLIB_PATH%\lib\zlib.lib
set CMAKE_OPT=%CMAKE_OPT% -D ZLIB_INCLUDE_DIR:PATH=%ZLIB_PATH%\include


:: -------------------------
:: OPENSSL_PATH can be set to an existing OpenSSL installation.
:: If not, we use vcpkg to install the library
::
echo.
echo ... OpenSSL
if "%OPENSSL_PATH%"=="" (
    %VCPKG% install openssl
    set OPENSSL_PATH=%VCPKG_INSTALL%
)
set CMAKE_OPT=%CMAKE_OPT% -D OPENSSL_ROOT_DIR:PATH=%OPENSSL_PATH%
set CMAKE_OPT=%CMAKE_OPT% -D OPENSSL_USE_STATIC_LIBS:BOOL=TRUE
set CMAKE_OPT=%CMAKE_OPT% -D OPENSSL_MSVC_STATIC_RT:BOOL=TRUE

echo.
echo.
echo --- Configuring CMake...

:: -----  Target Visual Studio 2017 64bits -----
set CMAKE_OPT=%CMAKE_OPT% -G"Visual Studio 15 2017 Win64" 

:: Policy CMP0074: find_package uses <PackageName>_ROOT variables.
set CMAKE_OPT=%CMAKE_OPT% -D CMAKE_POLICY_DEFAULT_CMP0074=NEW

:: -----  Disable some tool build -----
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_SERVER:BOOL=TRUE 
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

goto :eof


