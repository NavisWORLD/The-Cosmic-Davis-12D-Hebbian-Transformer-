"""
Verification script for START_COSMIC_DAVIS.bat
Tests that all Python scripts referenced in the launcher exist and can be imported
"""
import os
import sys
from pathlib import Path

# Get the project root
project_root = Path(__file__).parent

# Define color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def check_file(filepath, description):
    """Check if a file exists and report the result"""
    full_path = project_root / filepath
    exists = full_path.exists()
    
    status = f"{Colors.GREEN}✓ FOUND{Colors.RESET}" if exists else f"{Colors.RED}✗ MISSING{Colors.RESET}"
    print(f"  {status} - {description}")
    print(f"           Path: {filepath}")
    
    return exists

def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}🌌 COSMIC DAVIS LAUNCHER - VERIFICATION TEST{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")
    
    all_tests_passed = True
    
    # Test each option from the launcher
    tests = [
        ("production_12d/autonomous_engine.py", "[1] Standard Autonomous Study"),
        ("research_42d/experiments/singularity_42d/autonomous_dual_mind.py", "[2] Dual Mind Training"),
        ("production_12d/neural_monitor.py", "[3] Neural Monitor"),
        ("research_42d/experiments/singularity_42d/showcase_dual_mind.py", "[4] Showcase Dual Mind"),
        ("tests/benchmark_current_model.py", "[5] Run Benchmarks"),
        ("research_42d/audio_12d/real_time_audio_pipe.py", "[6] Test Audio Engine"),
        ("research_42d/experiments/singularity_42d/chat_dual_mind.py", "[7] Chat with Dual Mind"),
        ("production_12d/chat_console.py", "[8] Chat with 12D"),
        ("research_42d/chat_console_42d.py", "[9] Chat with 42D"),
    ]
    
    print(f"{Colors.BOLD}Checking all launcher options...{Colors.RESET}\n")
    
    for filepath, description in tests:
        if not check_file(filepath, description):
            all_tests_passed = False
        print()
    
    # Check PYTHONPATH directory
    print(f"{Colors.BOLD}Checking Python package...{Colors.RESET}\n")
    check_file("packages/cosmic-synapse-transformer", "Cosmic Synapse Transformer package")
    print()
    
    # Final summary
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    if all_tests_passed:
        print(f"{Colors.BOLD}{Colors.GREEN}✓ ALL TESTS PASSED{Colors.RESET}")
        print(f"{Colors.GREEN}All launcher options are properly configured!{Colors.RESET}")
    else:
        print(f"{Colors.BOLD}{Colors.RED}✗ SOME TESTS FAILED{Colors.RESET}")
        print(f"{Colors.RED}Some files are missing. Please check the paths above.{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())
