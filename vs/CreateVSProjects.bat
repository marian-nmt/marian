@echo off
setlocal

set ROOT=%~dp0
set MARIAN_ROOT=%ROOT%..

call CheckDeps.bat
if errorlevel 1 exit /b 1


echo.
echo --- Configuring CMake...

set CMAKE_OPT=

:: ----- Configure OpenSLL dep
set CMAKE_OPT=%CMAKE_OPT% -D OPENSSL_USE_STATIC_LIBS:BOOL=TRUE
set CMAKE_OPT=%CMAKE_OPT% -D OPENSSL_MSVC_STATIC_RT:BOOL=TRUE

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

popd

exit /b 0