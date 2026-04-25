@echo off
echo ============================================================
echo 🌌 STARTING AUTONOMOUS STUDY (THE BRAIN)
echo ============================================================
echo.

:: Set the path to the core package
set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer

:: Run the study script
python research_42d/experiments/singularity_42d/autonomous_study.py

pause
