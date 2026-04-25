@echo off
title CosmoSynapse Installer
color 0D
echo.
echo  ============================================
echo   COSMOSYNAPSE INSTALLER
echo   Cybernetic Bio Resonance Core v1.0.0
echo  ============================================
echo.
echo  Starting installation...
echo.

python install.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  Installation encountered errors.
    echo  Check the output above for details.
)

echo.
pause
