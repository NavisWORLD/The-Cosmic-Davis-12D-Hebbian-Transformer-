#!/usr/bin/env python3
"""
Internal Dimension AI - Easy Interface for Researchers

This script provides a simple menu-driven interface for running experiments
without needing programming knowledge.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import time


def clear_screen():
    """Clear the terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    """Print the application header"""
    print("=" * 75)
    print("  INTERNAL DIMENSION AI - RESEARCHER INTERFACE")
    print("=" * 75)
    print()


def print_menu():
    """Print the main menu"""
    print("What would you like to do?\n")
    print("  QUICK DEMOS (recommended for first-time users):")
    print("    1. Run Quick Demo (5 minutes) - Train a simple agent")
    print("    2. Run Baseline Comparison (15 minutes) - Compare IDN vs Standard")
    print("    3. Run Curiosity Demo (10 minutes) - Test curiosity-driven learning")
    print()
    print("  RESEARCH EXPERIMENTS (for comprehensive analysis):")
    print("    4. Dimensional Scaling Study - Test different internal dimensions")
    print("    5. Emergence Timeline Analysis - Track meta-awareness emergence")
    print("    6. Ablation Study - Test component contributions")
    print()
    print("  ANALYSIS TOOLS:")
    print("    7. Visualize Existing Results")
    print("    8. Evaluate Consciousness Metrics")
    print()
    print("  OTHER:")
    print("    9. Check System Requirements")
    print("    0. Exit")
    print()


def check_requirements():
    """Check if all requirements are installed"""
    print("\nChecking system requirements...")
    print("-" * 75)

    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}", end="")
    if sys.version_info >= (3, 8):
        print(" ✓")
    else:
        print(" ✗ (need 3.8+)")
        return False

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU: Available ({torch.cuda.get_device_name(0)}) ✓")
    else:
        print("GPU: Not available (will use CPU - slower but works)")

    # Check key packages
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'scikit-learn'),
        ('tqdm', 'tqdm'),
    ]

    all_good = True
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"{name:20s} ✓")
        except ImportError:
            print(f"{name:20s} ✗ MISSING")
            all_good = False

    print("-" * 75)
    if all_good:
        print("\n✓ All requirements satisfied!\n")
        return True
    else:
        print("\n✗ Some packages are missing. Please run: ./setup.sh\n")
        return False


def run_quick_demo():
    """Run the quick demo"""
    print("\n" + "=" * 75)
    print("  QUICK DEMO - Training Internal Dimension Network")
    print("=" * 75)
    print()
    print("This demo will:")
    print("  • Train an agent on a simple gridworld task")
    print("  • Show consciousness metrics evolution")
    print("  • Generate visualizations")
    print("  • Take approximately 5 minutes")
    print()

    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        return

    print("\nStarting demo...\n")
    os.system('python examples/01_quick_demo.py')

    print("\n" + "=" * 75)
    print("Demo complete! Check outputs/quick_demo/ for results.")
    print("=" * 75)
    input("\nPress ENTER to return to menu...")


def run_baseline_comparison():
    """Run baseline comparison"""
    print("\n" + "=" * 75)
    print("  BASELINE COMPARISON - IDN vs Standard PPO")
    print("=" * 75)
    print()
    print("This experiment will:")
    print("  • Train an Internal Dimension Network")
    print("  • Train a standard PPO baseline")
    print("  • Compare their performance and metrics")
    print("  • Take approximately 15 minutes")
    print()

    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        return

    print("\nStarting experiment...\n")
    os.system('python examples/02_baseline_comparison.py')

    print("\n" + "=" * 75)
    print("Experiment complete! Check outputs/baseline_comparison/ for results.")
    print("=" * 75)
    input("\nPress ENTER to return to menu...")


def run_curiosity_demo():
    """Run curiosity demo"""
    print("\n" + "=" * 75)
    print("  CURIOSITY DEMO - Curiosity-Driven Learning")
    print("=" * 75)
    print()
    print("This demo will:")
    print("  • Train agent with curiosity-driven exploration")
    print("  • Show how internal dimensions enable curiosity")
    print("  • Take approximately 10 minutes")
    print()

    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        return

    print("\nStarting demo...\n")
    os.system('python examples/03_curiosity_demo.py')

    print("\n" + "=" * 75)
    print("Demo complete! Check outputs/curiosity_demo/ for results.")
    print("=" * 75)
    input("\nPress ENTER to return to menu...")


def run_dimensional_scaling():
    """Run dimensional scaling experiment"""
    print("\n" + "=" * 75)
    print("  DIMENSIONAL SCALING STUDY")
    print("=" * 75)
    print()
    print("This experiment will:")
    print("  • Test 7 different internal dimension sizes (0, 4, 8, 12, 24, 64, 128)")
    print("  • Run 5 replications of each (35 trials total)")
    print("  • 300 episodes per trial")
    print("  • Perform statistical analysis")
    print()
    print("⚠️  WARNING: This is a comprehensive experiment!")
    print("  • Estimated time: 4-8 hours (depending on your hardware)")
    print("  • Requires significant computational resources")
    print()

    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        return

    print("\nStarting experiment (this will take several hours)...\n")
    os.system('python scripts/run_experiment.py --config configs/experiments/dimensional_scaling.yaml')

    print("\n" + "=" * 75)
    print("Experiment complete! Check outputs/experiments/dimensional_scaling/")
    print("=" * 75)
    input("\nPress ENTER to return to menu...")


def run_emergence_timeline():
    """Run emergence timeline experiment"""
    print("\n" + "=" * 75)
    print("  EMERGENCE TIMELINE ANALYSIS")
    print("=" * 75)
    print()
    print("This experiment will:")
    print("  • Track meta-awareness emergence over 1000 episodes")
    print("  • Run 10 replications with different seeds")
    print("  • Identify critical transition points")
    print("  • Perform statistical emergence tests")
    print()
    print("⚠️  WARNING: This is a long experiment!")
    print("  • Estimated time: 6-10 hours")
    print("  • 10,000 total episodes")
    print()

    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        return

    print("\nStarting experiment (this will take many hours)...\n")
    os.system('python scripts/run_experiment.py --config configs/experiments/emergence_timeline.yaml')

    print("\n" + "=" * 75)
    print("Experiment complete! Check outputs/experiments/emergence_timeline/")
    print("=" * 75)
    input("\nPress ENTER to return to menu...")


def run_ablation_study():
    """Run ablation study"""
    print("\n" + "=" * 75)
    print("  ABLATION STUDY - Component Analysis")
    print("=" * 75)
    print()
    print("This experiment will:")
    print("  • Test 6 different configurations")
    print("  • Test on 3 different environments")
    print("  • Run 5 replications each (90 trials total)")
    print("  • Identify which components are necessary")
    print()
    print("⚠️  WARNING: This is the longest experiment!")
    print("  • Estimated time: 10-15 hours")
    print("  • 90 total trials × 400 episodes each")
    print()

    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        return

    print("\nStarting experiment (this will take a very long time)...\n")
    os.system('python scripts/run_experiment.py --config configs/experiments/ablation_study.yaml')

    print("\n" + "=" * 75)
    print("Experiment complete! Check outputs/experiments/ablation_study/")
    print("=" * 75)
    input("\nPress ENTER to return to menu...")


def visualize_results():
    """Visualize existing results"""
    print("\n" + "=" * 75)
    print("  VISUALIZE RESULTS")
    print("=" * 75)
    print()
    print("Available result directories:")
    print()

    # Check for result directories
    outputs_dir = Path('outputs')
    if not outputs_dir.exists():
        print("No results found yet. Run an experiment first!")
        input("\nPress ENTER to return to menu...")
        return

    result_dirs = []
    for item in outputs_dir.iterdir():
        if item.is_dir():
            result_dirs.append(item)

    if not result_dirs:
        print("No results found yet. Run an experiment first!")
        input("\nPress ENTER to return to menu...")
        return

    for idx, dir_path in enumerate(result_dirs, 1):
        print(f"  {idx}. {dir_path.name}")

    print()
    choice = input("Select directory to visualize (number): ").strip()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(result_dirs):
            result_path = result_dirs[idx]
            print(f"\nVisualizing results from: {result_path}\n")
            os.system(f'python scripts/visualize_results.py --results_dir {result_path}')
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")

    input("\nPress ENTER to return to menu...")


def evaluate_consciousness():
    """Evaluate consciousness metrics"""
    print("\n" + "=" * 75)
    print("  EVALUATE CONSCIOUSNESS METRICS")
    print("=" * 75)
    print()
    print("This tool evaluates consciousness metrics for a trained model.")
    print()

    # Look for checkpoint files
    checkpoint_dirs = list(Path('checkpoints').rglob('*.pt')) if Path('checkpoints').exists() else []

    if not checkpoint_dirs:
        print("No checkpoints found. Train a model first!")
        input("\nPress ENTER to return to menu...")
        return

    print("Available checkpoints:")
    for idx, ckpt in enumerate(checkpoint_dirs[:10], 1):  # Show first 10
        print(f"  {idx}. {ckpt.relative_to('checkpoints')}")

    print()
    choice = input("Select checkpoint (number) or press ENTER to skip: ").strip()

    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoint_dirs):
                ckpt_path = checkpoint_dirs[idx]
                print(f"\nEvaluating: {ckpt_path}\n")
                os.system(f'python scripts/evaluate_consciousness.py --checkpoint {ckpt_path}')
        except (ValueError, IndexError):
            print("Invalid selection.")

    input("\nPress ENTER to return to menu...")


def main():
    """Main application loop"""
    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input("Enter your choice (0-9): ").strip()

        if choice == '1':
            run_quick_demo()
        elif choice == '2':
            run_baseline_comparison()
        elif choice == '3':
            run_curiosity_demo()
        elif choice == '4':
            run_dimensional_scaling()
        elif choice == '5':
            run_emergence_timeline()
        elif choice == '6':
            run_ablation_study()
        elif choice == '7':
            visualize_results()
        elif choice == '8':
            evaluate_consciousness()
        elif choice == '9':
            check_requirements()
            input("\nPress ENTER to return to menu...")
        elif choice == '0':
            print("\nThank you for using Internal Dimension AI!")
            print("For questions or issues, please see README.md\n")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please try again.")
            time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
