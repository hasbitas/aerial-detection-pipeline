@echo off
setlocal
cd /d "%~dp0"
echo [Setup] Checking Python 3.11 or newer...
if exist ".venv\Scripts\python.exe" goto existing
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 goto use_launcher
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto missing_python
set "PYTHON=python"
goto ready
:use_launcher
set "PYTHON=py -3"
goto ready
:existing
set "PYTHON=.venv\Scripts\python.exe"
%PYTHON% -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 goto missing_python
:ready
if not exist requirements.txt goto missing_files
if not exist demo\requirements.txt goto missing_files
if /i "%~1"=="--check" goto check_done
if exist ".venv\Scripts\python.exe" goto install
echo [Setup] Creating .venv...
%PYTHON% -m venv .venv
if errorlevel 1 goto failed
:install
echo [Setup] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
echo [Setup] Installing project and browser demo packages. Downloads may take a while...
".venv\Scripts\python.exe" -m pip install -r requirements.txt -r demo\requirements.txt
if errorlevel 1 goto failed
echo [Setup] Setup complete. Next, double-click start.bat.
pause
exit /b 0
:check_done
echo [Setup] Prerequisites OK. Dry check only; nothing installed.
exit /b 0
:missing_python
echo [Setup] Python 3.11+ is required. Install it from https://www.python.org/downloads/
echo [Setup] Tick Add Python to PATH, reopen this window, and try again.
goto failed
:missing_files
echo [Setup] Requirements files are missing. Extract the whole project ZIP first.
:failed
echo [Setup] Setup failed. Read the error above, fix it, and run setup.bat again.
if /i not "%~1"=="--check" pause
exit /b 1
