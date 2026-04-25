@echo off
chcp 65001 >nul
title 🌌 COSMIC DAVIS OPERATING SYSTEM
mode con: cols=80 lines=30

:MENU
cls
echo.
echo   ╔══════════════════════════════════════════════════════════════════════╗
echo   ║                🌌 THE COSMIC DAVIS: 12D HEBBIAN TRANSFORMER          ║
echo   ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo   [ TRAINING ] ───────────────────────────────────────────────────────────
echo.
echo      [1] 🚀 Production Training (Continuous)
echo      [2] 🧠 Evolutionary Instruction Tuner (RLHF Polish)
echo          » Rapid fine-tuning on Q^&A pairs
echo          » Instruction tuning for conversational fluency
echo      [3] 🌌 Dual Mind Evolution (Synchronized Training)
echo.
echo   [ CHAT MODES ] ─────────────────────────────────────────────────────────
echo.
echo      [4] 🔥 UNHINGED SENSORY HYBRID (Full Power)
echo          » GPT-2 + 12D + Audio + Emotion + NO LIMITS
echo      [5] ⭐ GPT-2 Hybrid Chat (Pre-trained Baseline)
echo      [6] 💬 Dual Mind Chat (Custom Trained 12D + 42D)
echo.
echo   [ SYSTEM TOOLS ] ───────────────────────────────────────────────────────
echo.
echo      [7] 👁️  Neural Monitor (Real-time Cognition Dashboard)
echo      [8] 🎵 Audio Engine Test
echo      [9] 📥 Download Weights
echo      [0] 🧪 System Benchmarks
echo.
echo      [X] ❌ EXIT SYSTEM
echo.
echo   ────────────────────────────────────────────────────────────────────────
echo.
set /p choice="  > Select System Core: "

if /i "%choice%"=="1" goto TRAIN_PRODUCTION
if /i "%choice%"=="2" goto TRAIN_EVOLUTIONARY
if /i "%choice%"=="3" goto TRAIN_DUAL
if /i "%choice%"=="4" goto CHAT_UNHINGED
if /i "%choice%"=="5" goto CHAT_GPT2_HYBRID
if /i "%choice%"=="6" goto CHAT_DUAL
if /i "%choice%"=="7" goto MONITOR
if /i "%choice%"=="8" goto AUDIO_TEST
if /i "%choice%"=="9" goto DOWNLOAD_WEIGHTS
if /i "%choice%"=="0" goto BENCHMARK
if /i "%choice%"=="X" exit

echo.
echo Invalid choice. Please try again.
pause
goto MENU

:TRAIN_PRODUCTION
echo.
echo >> Launching PRODUCTION TRAINING ENGINE in new window...
echo    (Optimized learning with gradient accumulation)
start "Cosmic Davis Production" cmd /k "title PRODUCTION TRAINING && set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer && python app/core/production_engine.py"
goto MENU

:TRAIN_EVOLUTIONARY
echo.
echo >> Launching EVOLUTIONARY INSTRUCTION TUNER (RLHF)...
set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer
python app/core/evolutionary_tuner.py
pause
goto MENU

:TRAIN_DUAL
echo.
echo >> Launching Dual Mind Evolution System in new window...
start "Dual Mind Evolution" cmd /k "title DUAL MIND EVOLUTION && set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer && python app/modules/dual_mind/autonomous_dual_mind.py"
goto MENU

:CHAT_UNHINGED
cls
echo.
echo ============================================================
echo 🔥 COSMIC DAVIS - UNHINGED SENSORY HYBRID
echo    GPT-2 + 12D + Audio + Emotion + NO LIMITS
echo ============================================================
echo.
set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer
python cosmic_unhinged_hybrid.py
echo.
echo [Exited Chat]
pause
goto MENU

:CHAT_GPT2_HYBRID
cls
echo.
echo ============================================================
echo ⭐ COSMIC DAVIS - GPT-2 HYBRID CHAT
echo    Using pre-trained GPT-2 + 12D Cosmic Enhancements
echo ============================================================
echo.
set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer
python cosmic_gpt2_hybrid.py
echo.
echo [Exited Chat]
pause
goto MENU

:CHAT_DUAL
cls
echo.
echo ============================================================
echo 💬 DUAL MIND CHAT (Custom Trained 12D + 42D)
echo ============================================================
echo.
set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer
python app/ui/chat_dual.py
echo.
echo [Exited Chat]
pause
goto MENU

:MONITOR
echo.
echo >> Launching Neural Monitor in new window...
start "Neural Monitor" cmd /k "title NEURAL MONITOR && set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer && python app/ui/monitor.py"
goto MENU

:BENCHMARK
echo.
echo >> Running Benchmarks...
set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer
python app/tools/benchmark.py
pause
goto MENU

:AUDIO_TEST
echo.
echo >> Testing Audio Engine...
set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer
python app/modules/audio/real_time_audio_pipe.py
pause
goto MENU

:DOWNLOAD_WEIGHTS
echo.
echo >> Downloading Pre-trained Weights...
python download_pretrained_weights.py
pause
goto MENU

