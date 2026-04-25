@echo off
title DUAL MIND EVOLUTION - Training
cls
echo ============================================================
echo 🧠 DUAL MIND EVOLUTIONARY SYSTEM
echo ============================================================
echo.
echo 12D (Past/Grounded) ^<-^> 42D (Future/Abstract)
echo.
echo This will start autonomous training where the two minds:
echo  - Learn independently from texts
echo  - Debate every 10 iterations
echo  - Auto-save every 100 iterations
echo  - Build infinite memory
echo.
echo ============================================================
pause

set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer
python research_42d/experiments/singularity_42d/autonomous_dual_mind.py

pause
