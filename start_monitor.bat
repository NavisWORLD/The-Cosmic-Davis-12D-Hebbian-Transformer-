@echo off
rem Set the Python path to include the core transformer package
set "PYTHONPATH=%~dp0\packages\cosmic-synapse-transformer"

rem Run the neural monitor dashboard
python production_12d/neural_monitor.py

rem Keep the window open after the script exits
pause
