@echo off
setlocal

set ROOT=%~dp0
set BUILD_ROOT=%ROOT%build-vs

cmake --build %BUILD_ROOT% --config Release 
