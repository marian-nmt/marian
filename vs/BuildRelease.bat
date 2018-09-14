@echo off
setlocal

set ROOT=%~dp0
set BUILD_ROOT=%ROOT%build-vs

call CreateVSProjects.bat
if errorlevel 1 exit /b 1

cmake --build %BUILD_ROOT% --config Release 

exit /b 0