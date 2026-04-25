"""
Quick verification script for START_COSMIC_DAVIS.bat
"""
import os
from pathlib import Path

project_root = Path(__file__).parent

# Files to check
files_to_check = [
    "production_12d/autonomous_engine.py",
    "research_42d/experiments/singularity_42d/autonomous_dual_mind.py",
    "production_12d/neural_monitor.py",
    "research_42d/experiments/singularity_42d/showcase_dual_mind.py",
    "tests/benchmark_current_model.py",
    "research_42d/audio_12d/real_time_audio_pipe.py",
    "research_42d/experiments/singularity_42d/chat_dual_mind.py",
    "production_12d/chat_console.py",
    "research_42d/chat_console_42d.py",
    "packages/cosmic-synapse-transformer",
]

print("\nCOSMIC DAVIS LAUNCHER VERIFICATION")
print("=" * 60)

all_ok = True
for filepath in files_to_check:
    full_path = project_root / filepath
    exists = full_path.exists()
    status = "OK" if exists else "MISSING"
    print(f"{status:8} {filepath}")
    if not exists:
        all_ok = False

print("=" * 60)
if all_ok:
    print("RESULT: All files found - Launcher is ready!")
else:
    print("RESULT: Some files missing - Check paths above")
print()
