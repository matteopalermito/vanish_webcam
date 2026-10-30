@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=.venv-local"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [VanishCam] Creating the local Python environment...
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv "%VENV_DIR%"
  ) else (
    python -m venv "%VENV_DIR%"
  )
  if errorlevel 1 goto :error
)

"%VENV_PY%" -c "import cv2, mediapipe, numpy, pyvirtualcam, pygame" >nul 2>nul
if errorlevel 1 (
  echo [VanishCam] Installing required components. Internet is needed only this time...
  "%VENV_PY%" -m pip install --upgrade pip
  if errorlevel 1 goto :error
  "%VENV_PY%" -m pip install -r "requirements.txt"
  if errorlevel 1 goto :error
)

echo [VanishCam] Starting the webcam...
"%VENV_PY%" "vanish_cam.py" %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Unable to start VanishCam. See README.md for common solutions.
pause
exit /b 1
