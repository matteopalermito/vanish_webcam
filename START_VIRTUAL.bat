@echo off
call "%~dp0START_LOCAL.bat" --virtual-camera %*
exit /b %errorlevel%
