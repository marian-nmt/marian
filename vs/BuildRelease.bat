::
:: Usage: BuildRelease.bat [<build-directory>=.\build]
::
:: This script runs the dependency checks, generate the projects/makefiles and then 
:: build the project in Release configuration.
::
::
@echo off
setlocal

set ROOT=%~dp0
set MARIAN_ROOT=%ROOT%..

set BUILD_ROOT=%1
if "%BUILD_ROOT%"=="" set BUILD_ROOT=%ROOT%build

call CreateVSProjects.bat %BUILD_ROOT%
if errorlevel 1 exit /b 1

cmake --build %BUILD_ROOT% --config Release

exit /b 0