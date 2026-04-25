@echo off
title DUAL MIND CHAT
cls
echo ============================================================
echo 🌌 DUAL MIND CHAT INTERFACE
echo ============================================================
echo.
echo Talk to both Past (12D) and Future (42D) minds
echo Watch them debate and form consensus
echo.
echo ============================================================
pause

set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer
python research_42d/experiments/singularity_42d/chat_dual_mind.py

pause
