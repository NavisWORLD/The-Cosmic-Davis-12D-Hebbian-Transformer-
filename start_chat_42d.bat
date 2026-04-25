@echo off
echo ============================================================
echo 🌌 STARTING 42D HYPER CONSOLE (THE MIND)
echo ============================================================
echo.

:: Set the path to the core package
set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer

:: Run the chat script
python research_42d/chat_console_42d.py

pause
