@echo off
setlocal
set WEB_HOST=127.0.0.1
set WEB_PORT=8000
echo Starting PyCinemetrics WebUI on %WEB_HOST%:%WEB_PORT%
python -m pip show flask >NUL 2>&1
if errorlevel 1 (
  echo Flask not found. Install with: pip install -r webui\requirements.txt
)
python src\webserver.py
endlocal
