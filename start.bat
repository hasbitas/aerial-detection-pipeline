@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing_setup
set "GRADIO_SERVER_NAME=127.0.0.1"
set "GRADIO_SERVER_PORT=7860"
echo [Start] Starting the demo at http://127.0.0.1:7860
echo [Start] Keep this window open. Press Ctrl+C or close it to stop.
echo [Start] Model weights download on the first Analyze click if missing.
".venv\Scripts\python.exe" -u demo\app.py --device cpu --open-browser
if errorlevel 1 goto failed
exit /b 0
:missing_setup
echo [Start] Setup is missing. Run setup.bat first.
goto error_exit
:failed
echo [Start] The demo stopped with an error. Read the message above.
echo [Start] Check setup completed and port 7860 is free, then try again.
:error_exit
pause
exit /b 1
