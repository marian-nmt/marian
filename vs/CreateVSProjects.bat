::
:: Usage: CreateVSProjects.bat [<build-directory>=.\build]
::
:: This script runs the dependency checks, then invokes CMake with the right parameters to create 
:: the solutions for Visual Studio.
::
:: Run this script only if you have a previous version of Visual Studio (2015 and below), or if you 
:: don't want to use the built-in CMake integration.
::
:: You may want to change the generator target to fit your installation. By default, the target is
:: "Visual Studio 15 2017 Win64". Run `cmake --help` for a list of supported targets.
::
:: Note: You don't need to run this script if you want to use Visual Studio with built-in support for CMake
:: (only available for VS 2017+)
::
::
@echo off
setlocal

set ROOT=%~dp0
set MARIAN_ROOT=%ROOT%..

set BUILD_ROOT=%1
if "%BUILD_ROOT%"=="" set BUILD_ROOT=%ROOT%build

set GENERATOR_TARGET="Visual Studio 15 2017 Win64"

call CheckOrInstallDeps.bat
if errorlevel 1 exit /b 1


echo.
echo --- Configuring CMake...

set CMAKE_OPT=

:: ----- Configure OpenSLL dep
set CMAKE_OPT=%CMAKE_OPT% -D OPENSSL_USE_STATIC_LIBS:BOOL=TRUE
set CMAKE_OPT=%CMAKE_OPT% -D OPENSSL_MSVC_STATIC_RT:BOOL=TRUE

:: -----  Target Visual Studio 2017 64bits -----
set CMAKE_OPT=%CMAKE_OPT% -G %GENERATOR_TARGET%

:: Policy CMP0074: find_package uses <PackageName>_ROOT variables.
set CMAKE_OPT=%CMAKE_OPT% -D CMAKE_POLICY_DEFAULT_CMP0074=NEW

:: -----  Disable some tool build -----
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_EXAMPLES:BOOL=FALSE 
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_TESTS:BOOL=FALSE 
set CMAKE_OPT=%CMAKE_OPT% -D USE_MPI:BOOL=FALSE
set CMAKE_OPT=%CMAKE_OPT% -D USE_CUDNN:BOOL=FALSE

:: -----  Enable certain options -----
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_SERVER:BOOL=TRUE
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_CPU:BOOL=TRUE 
set CMAKE_OPT=%CMAKE_OPT% -D COMPILE_CUDA:BOOL=TRUE
set CMAKE_OPT=%CMAKE_OPT% -D USE_SENTENCEPIECE:BOOL=ON

:: -----  Not supported on Windows yet -----
set CMAKE_OPT=%CMAKE_OPT% -D USE_NCCL:BOOL=FALSE


echo.
echo.
echo --------------------------------------------------
echo CMAKE configuration flags:
echo %CMAKE_OPT%
echo --------------------------------------------------
echo.
echo.

if not exist %BUILD_ROOT% mkdir %BUILD_ROOT%
pushd %BUILD_ROOT%


echo.
echo --- Creating Visual Studio projects...
cmake %CMAKE_OPT% %MARIAN_ROOT%

popd

exit /b 0
