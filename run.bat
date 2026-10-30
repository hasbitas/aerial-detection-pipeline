@echo off
setlocal
cd /d "%~dp0"
rem Run with models/yolo26s-aerial.pt; pass all pipeline arguments through.
rem Example: run.bat --frames path\to\frames --no-display --device cpu
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" run_pipeline.py %*
exit /b %ERRORLEVEL%
