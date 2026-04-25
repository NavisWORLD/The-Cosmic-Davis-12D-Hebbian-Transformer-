"""
Thorough verification of START_COSMIC_DAVIS.bat file paths
"""
import os
from pathlib import Path

project_root = Path(__file__).parent

# Exact paths from the batch file
paths_to_verify = {
    "[1] Training": "production_12d/autonomous_engine.py",
    "[2] Dual Mind Training": "research_42d/experiments/singularity_42d/autonomous_dual_mind.py",
    "[3] Monitor": "production_12d/neural_monitor.py",
    "[4] Showcase": "research_42d/experiments/singularity_42d/showcase_dual_mind.py",
    "[5] Benchmark": "tests/benchmark_current_model.py",
    "[6] Audio Test": "research_42d/audio_12d/real_time_audio_pipe.py",
    "[7] Chat Dual": "research_42d/experiments/singularity_42d/chat_dual_mind.py",
    "[8] Chat 12D": "production_12d/chat_console.py",
    "[9] Chat 42D": "research_42d/chat_console_42d.py",
    "PYTHONPATH": "packages/cosmic-synapse-transformer",
}

print("\n" + "=" * 70)
print("COSMIC DAVIS LAUNCHER - COMPLETE FILE PATH VERIFICATION")
print("=" * 70 + "\n")

all_found = True
missing = []

for label, filepath in paths_to_verify.items():
    full_path = project_root / filepath
    exists = full_path.exists()
    
    status = "[OK]     " if exists else "[MISSING]"
    print(f"{status} {label:25} {filepath}")
    
    if not exists:
        all_found = False
        missing.append((label, filepath))

print("\n" + "=" * 70)

if all_found:
    print("✓ SUCCESS: All file paths are correct!")
    print("\nThe launcher is ready to use.")
else:
    print("✗ ERRORS FOUND: The following files are missing:\n")
    for label, filepath in missing:
        print(f"  - {label}: {filepath}")
    print("\nThese paths need to be corrected in START_COSMIC_DAVIS.bat")

print("=" * 70 + "\n")
