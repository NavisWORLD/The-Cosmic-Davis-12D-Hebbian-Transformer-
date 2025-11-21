@echo off
REM Quick start script for Windows - activates environment and runs the easy interface

cd /d "%~dp0"

if not exist venv (
    echo Virtual environment not found. Running setup first...
    call setup_windows.bat
)

call venv\Scripts\activate.bat
python run_easy.py
pause
