@echo off
setlocal

rem --------------------------------------------------------------------
rem PyCinemetrics WebUI launcher
rem  - Creates an isolated virtualenv (.webui-venv) the first time
rem  - Installs/updates dependencies from webui\requirements.txt
rem  - Starts src\webserver.py with consistent interpreter
rem
rem Customize with:
rem   WEBUI_PYTHON      -> absolute path to python.exe to use
rem   WEBUI_SKIP_INSTALL -> set to 1 to skip dependency check
rem   WEB_HOST / WEB_PORT -> override host/port
rem --------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "REQ_FILE=%SCRIPT_DIR%webui\requirements.txt"
set "SERVER_SCRIPT=%SCRIPT_DIR%src\webserver.py"
set "VENV_DIR=%SCRIPT_DIR%.webui-venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not defined WEB_HOST set "WEB_HOST=127.0.0.1"
if not defined WEB_PORT set "WEB_PORT=8000"

set "PYTHON_CMD="
set "USING_ISOLATED_ENV="

if defined WEBUI_PYTHON (
  if exist "%WEBUI_PYTHON%" (
    set "PYTHON_CMD=%WEBUI_PYTHON%"
  ) else (
    echo WEBUI_PYTHON is set but "%WEBUI_PYTHON%" does not exist.
    exit /b 1
  )
) else (
  if not exist "%VENV_PY%" (
    echo [PyCinemetrics] Creating dedicated WebUI virtual environment...
    python -m venv "%VENV_DIR%" >NUL 2>&1
    if errorlevel 1 (
      py -3 -m venv "%VENV_DIR%" >NUL 2>&1
    )
    if not exist "%VENV_PY%" (
      echo Failed to create .webui-venv. Install Python 3.8+ or set WEBUI_PYTHON.
      exit /b 1
    )
  )
  set "PYTHON_CMD=%VENV_PY%"
)

if /I "%PYTHON_CMD%"=="%VENV_PY%" set "USING_ISOLATED_ENV=1"

if defined WEBUI_SKIP_INSTALL (
  echo [PyCinemetrics] WEBUI_SKIP_INSTALL detected, skipping dependency install.
) else (
  call :install_dependencies
  if errorlevel 1 exit /b 1
)

echo(
echo [PyCinemetrics] Starting WebUI on %WEB_HOST%:%WEB_PORT%
set "PYTHONPATH=%SCRIPT_DIR%src;%PYTHONPATH%"
"%PYTHON_CMD%" "%SERVER_SCRIPT%"
exit /b %errorlevel%


:install_dependencies
if not exist "%REQ_FILE%" (
  echo Requirements file not found: "%REQ_FILE%"
  exit /b 1
)

echo [PyCinemetrics] Installing/Updating WebUI dependencies...
if defined USING_ISOLATED_ENV "%PYTHON_CMD%" -m pip install --upgrade pip >NUL
"%PYTHON_CMD%" -m pip install --disable-pip-version-check -r "%REQ_FILE%"
"%PYTHON_CMD%" -m pip install --disable-pip-version-check --no-deps face-recognition==1.3.0
exit /b %errorlevel%
