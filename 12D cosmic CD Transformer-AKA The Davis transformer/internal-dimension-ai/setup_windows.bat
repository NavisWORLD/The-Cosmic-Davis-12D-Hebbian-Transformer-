@echo off
REM Internal Dimension AI - Windows Setup Script
REM For researchers without programming experience

echo =========================================================================
echo   INTERNAL DIMENSION AI - AUTOMATIC SETUP (WINDOWS)
echo =========================================================================
echo.
echo This script will:
echo   1. Check your Python version
echo   2. Create a virtual environment
echo   3. Install all required packages
echo   4. Verify everything is working
echo.
echo This may take 5-10 minutes depending on your internet speed.
echo.
pause

REM Step 1: Check Python
echo Step 1/4: Checking Python version...
echo -----------------------------------------------------------------------
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation!
    pause
    exit /b 1
)
echo.

REM Step 2: Create virtual environment
echo Step 2/4: Creating virtual environment...
echo -----------------------------------------------------------------------
if exist venv (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    echo Virtual environment created
)
echo.

REM Step 3: Install packages
echo Step 3/4: Installing packages (this may take several minutes)...
echo -----------------------------------------------------------------------
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo All packages installed
echo.

REM Step 4: Verify
echo Step 4/4: Verifying installation...
echo -----------------------------------------------------------------------
python -c "import torch; import numpy; import pandas; print('All packages verified!')"
if errorlevel 1 (
    echo.
    echo Setup encountered errors. Please check the output above.
    pause
    exit /b 1
)

echo.
echo =========================================================================
echo   SETUP COMPLETE!
echo =========================================================================
echo.
echo Everything is ready to use!
echo.
echo To get started:
echo   1. Double-click: start_windows.bat
echo   2. Or run: python run_easy.py
echo.
pause
