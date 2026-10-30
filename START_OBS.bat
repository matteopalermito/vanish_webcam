@echo off
call "%~dp0START_LOCAL.bat" --no-virtual-camera %*
exit /b %errorlevel%
