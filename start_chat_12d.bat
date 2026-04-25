@echo off
echo ============================================================
echo 🌌 STARTING 12D CHAT CONSOLE (THE VOICE)
echo ============================================================
echo.

:: Set the path to the core package
set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer

:: Run the chat script
python production_12d/chat_console.py

pause
