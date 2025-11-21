#!/usr/bin/env python3
"""
Visualization Tool for Internal Dimension AI Experiments

Standalone CLI tool to generate plots from saved experiment logs.
Supports multiple visualization types including learning curves, internal
dimensions, consciousness metrics, and behavioral analysis.

Usage:
    python scripts/visualize_results.py --log data/experiment_01/
    python scripts/visualize_results.py --log data/ --type learning_curves
    python scripts/visualize_results.py --log data/ --type consciousness --output plots/

Examples:
    # Generate all plots from experiment log
    python scripts/visualize_results.py --log data/experiment_01/

    # Generate specific plot type
    python scripts/visualize_results.py --log data/exp/ --type internal_dims

    # Save to specific output directory
    python scripts/visualize_results.py --log data/ --output results/plots/

    # Generate all plots with custom DPI
    python scripts/visualize_results.py --log data/ --dpi 300
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_log(log_dir: Path) -> Dict:
    """
    Load training log from JSON file.

    Args:
        log_dir: Directory containing training_log.json

    Returns:
        Training log dictionary

    Raises:
        FileNotFoundError: If training_log.json not found
    """
    log_file = log_dir / "training_log.json"

    if not log_file.exists():
        # Try to find any JSON file in directory
        json_files = list(log_dir.glob("*.json"))
        if json_files:
            log_file = json_files[0]
            logger.warning(f"training_log.json not found, using {log_file.name}")
        else:
            raise FileNotFoundError(f"No JSON log files found in {log_dir}")

    logger.info(f"Loading log from: {log_file}")

    with open(log_file) as f:
        return json.load(f)


def plot_learning_curves(log: Dict, output_dir: Optional[Path] = None):
    """
    Plot learning curves (rewards, loss, etc.).

    Args:
        log: Training log dictionary
        output_dir: Directory to save plots (if None, display only)
    """
    logger.info("Generating learning curves...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    epochs = log.get("epoch", [])
    rewards = log.get("reward", [])
    policy_loss = log.get("policy_loss", [])
    value_loss = log.get("value_loss", [])
    entropy = log.get("entropy", [])

    # Plot rewards
    if rewards:
        axes[0, 0].plot(epochs, rewards, linewidth=2, color='#2ecc71')
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Average Reward")
        axes[0, 0].set_title("Learning Curve: Rewards")
        axes[0, 0].grid(True, alpha=0.3)

    # Plot policy loss
    if policy_loss:
        axes[0, 1].plot(epochs, policy_loss, linewidth=2, color='#e74c3c')
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Policy Loss")
        axes[0, 1].set_title("Policy Loss")
        axes[0, 1].grid(True, alpha=0.3)

    # Plot value loss
    if value_loss:
        axes[1, 0].plot(epochs, value_loss, linewidth=2, color='#3498db')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Value Loss")
        axes[1, 0].set_title("Value Loss")
        axes[1, 0].grid(True, alpha=0.3)

    # Plot entropy
    if entropy:
        axes[1, 1].plot(epochs, entropy, linewidth=2, color='#9b59b6')
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Entropy")
        axes[1, 1].set_title("Policy Entropy")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / "learning_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_internal_dimensions(log: Dict, output_dir: Optional[Path] = None):
    """
    Plot internal dimension statistics (x₁₂, m₁₂).

    Args:
        log: Training log dictionary
        output_dir: Directory to save plots
    """
    logger.info("Generating internal dimension plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = log.get("epoch", [])
    x12_mean = log.get("x12_mean", [])
    x12_std = log.get("x12_std", [])
    m12_mean = log.get("m12_mean", [])
    m12_std = log.get("m12_std", [])

    # Plot x12 mean
    if x12_mean:
        x12_mean_arr = np.array(x12_mean)
        if x12_mean_arr.ndim == 2:
            # Plot mean across all dimensions
            mean_vals = x12_mean_arr.mean(axis=1)
            std_vals = x12_mean_arr.std(axis=1)
            axes[0, 0].plot(epochs, mean_vals, linewidth=2, color='#e67e22')
            axes[0, 0].fill_between(
                epochs,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.3,
                color='#e67e22'
            )
        else:
            axes[0, 0].plot(epochs, x12_mean, linewidth=2, color='#e67e22')

        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("x₁₂ Mean")
        axes[0, 0].set_title("Internal Dimension x₁₂ (Mean)")
        axes[0, 0].grid(True, alpha=0.3)

    # Plot x12 std
    if x12_std:
        x12_std_arr = np.array(x12_std)
        if x12_std_arr.ndim == 2:
            std_vals = x12_std_arr.mean(axis=1)
            axes[0, 1].plot(epochs, std_vals, linewidth=2, color='#d35400')
        else:
            axes[0, 1].plot(epochs, x12_std, linewidth=2, color='#d35400')

        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("x₁₂ Std Dev")
        axes[0, 1].set_title("Internal Dimension x₁₂ (Std Dev)")
        axes[0, 1].grid(True, alpha=0.3)

    # Plot m12 mean
    if m12_mean:
        m12_mean_arr = np.array(m12_mean)
        if m12_mean_arr.ndim == 2:
            mean_vals = m12_mean_arr.mean(axis=1)
            std_vals = m12_mean_arr.std(axis=1)
            axes[1, 0].plot(epochs, mean_vals, linewidth=2, color='#16a085')
            axes[1, 0].fill_between(
                epochs,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.3,
                color='#16a085'
            )
        else:
            axes[1, 0].plot(epochs, m12_mean, linewidth=2, color='#16a085')

        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("m₁₂ Mean")
        axes[1, 0].set_title("Meta-Awareness m₁₂ (Mean)")
        axes[1, 0].grid(True, alpha=0.3)

    # Plot m12 std
    if m12_std:
        m12_std_arr = np.array(m12_std)
        if m12_std_arr.ndim == 2:
            std_vals = m12_std_arr.mean(axis=1)
            axes[1, 1].plot(epochs, std_vals, linewidth=2, color='#138d75')
        else:
            axes[1, 1].plot(epochs, m12_std, linewidth=2, color='#138d75')

        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("m₁₂ Std Dev")
        axes[1, 1].set_title("Meta-Awareness m₁₂ (Std Dev)")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / "internal_dimensions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_consciousness_metrics(log: Dict, output_dir: Optional[Path] = None):
    """
    Plot consciousness metrics (R_ω, R_ψ, φ).

    Args:
        log: Training log dictionary
        output_dir: Directory to save plots
    """
    logger.info("Generating consciousness metrics plots...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    epochs = log.get("epoch", [])
    R_omega = log.get("R_omega", [])
    R_psi = log.get("R_psi", [])
    phi = log.get("phi", [])

    # Plot R_ω
    if R_omega:
        axes[0].plot(epochs, R_omega, linewidth=2, color='#8e44ad')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("R_ω")
        axes[0].set_title("Internal Dimension Richness (R_ω)")
        axes[0].grid(True, alpha=0.3)

    # Plot R_ψ
    if R_psi:
        axes[1].plot(epochs, R_psi, linewidth=2, color='#c0392b')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("R_ψ")
        axes[1].set_title("Phenomenal Binding (R_ψ)")
        axes[1].grid(True, alpha=0.3)

    # Plot φ
    if phi:
        axes[2].plot(epochs, phi, linewidth=2, color='#27ae60')
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("φ")
        axes[2].set_title("Integrated Information (φ)")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / "consciousness_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_intrinsic_rewards(log: Dict, output_dir: Optional[Path] = None):
    """
    Plot intrinsic reward statistics.

    Args:
        log: Training log dictionary
        output_dir: Directory to save plots
    """
    logger.info("Generating intrinsic reward plots...")

    intrinsic_reward = log.get("intrinsic_reward", [])
    if not intrinsic_reward:
        logger.warning("No intrinsic reward data found")
        return

    epochs = log.get("epoch", [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot intrinsic reward over time
    axes[0].plot(epochs, intrinsic_reward, linewidth=2, color='#f39c12')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Intrinsic Reward")
    axes[0].set_title("Intrinsic Reward Over Time")
    axes[0].grid(True, alpha=0.3)

    # Plot histogram
    axes[1].hist(intrinsic_reward, bins=50, color='#f39c12', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel("Intrinsic Reward")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Intrinsic Reward Distribution")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / "intrinsic_rewards.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_exploration(log: Dict, output_dir: Optional[Path] = None):
    """
    Plot exploration statistics.

    Args:
        log: Training log dictionary
        output_dir: Directory to save plots
    """
    logger.info("Generating exploration plots...")

    unique_states = log.get("unique_states_visited", [])
    entropy = log.get("entropy", [])

    if not unique_states and not entropy:
        logger.warning("No exploration data found")
        return

    epochs = log.get("epoch", [])

    if unique_states and entropy:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot unique states
        axes[0].plot(epochs, unique_states, linewidth=2, color='#1abc9c')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Unique States Visited")
        axes[0].set_title("Exploration: Unique States")
        axes[0].grid(True, alpha=0.3)

        # Plot entropy
        axes[1].plot(epochs, entropy, linewidth=2, color='#9b59b6')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Policy Entropy")
        axes[1].set_title("Exploration: Policy Entropy")
        axes[1].grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = unique_states if unique_states else entropy
        label = "Unique States Visited" if unique_states else "Policy Entropy"
        color = '#1abc9c' if unique_states else '#9b59b6'

        ax.plot(epochs, data, linewidth=2, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Exploration: {label}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / "exploration.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_summary(log: Dict, output_dir: Optional[Path] = None):
    """
    Plot comprehensive summary dashboard.

    Args:
        log: Training log dictionary
        output_dir: Directory to save plots
    """
    logger.info("Generating summary dashboard...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    epochs = log.get("epoch", [])

    # Reward
    ax1 = fig.add_subplot(gs[0, :2])
    if log.get("reward"):
        ax1.plot(epochs, log["reward"], linewidth=2, color='#2ecc71')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Reward")
        ax1.set_title("Average Reward", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

    # Consciousness metrics
    ax2 = fig.add_subplot(gs[0, 2])
    if log.get("R_omega") and log.get("R_psi") and log.get("phi"):
        ax2.plot(epochs, log["R_omega"], label='R_ω', linewidth=2)
        ax2.plot(epochs, log["R_psi"], label='R_ψ', linewidth=2)
        ax2.plot(epochs, log["phi"], label='φ', linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Value")
        ax2.set_title("Consciousness", fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Internal dimensions
    ax3 = fig.add_subplot(gs[1, 0])
    if log.get("x12_mean"):
        x12_mean = np.array(log["x12_mean"])
        if x12_mean.ndim == 2:
            x12_mean = x12_mean.mean(axis=1)
        ax3.plot(epochs, x12_mean, linewidth=2, color='#e67e22')
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("x₁₂ Mean")
        ax3.set_title("Internal Dim (x₁₂)", fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    if log.get("m12_mean"):
        m12_mean = np.array(log["m12_mean"])
        if m12_mean.ndim == 2:
            m12_mean = m12_mean.mean(axis=1)
        ax4.plot(epochs, m12_mean, linewidth=2, color='#16a085')
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("m₁₂ Mean")
        ax4.set_title("Meta-Awareness (m₁₂)", fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

    # Losses
    ax5 = fig.add_subplot(gs[1, 2])
    if log.get("policy_loss") and log.get("value_loss"):
        ax5.plot(epochs, log["policy_loss"], label='Policy', linewidth=2)
        ax5.plot(epochs, log["value_loss"], label='Value', linewidth=2)
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Loss")
        ax5.set_title("Training Losses", fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # Entropy
    ax6 = fig.add_subplot(gs[2, 0])
    if log.get("entropy"):
        ax6.plot(epochs, log["entropy"], linewidth=2, color='#9b59b6')
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("Entropy")
        ax6.set_title("Policy Entropy", fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

    # Intrinsic reward
    ax7 = fig.add_subplot(gs[2, 1])
    if log.get("intrinsic_reward"):
        ax7.plot(epochs, log["intrinsic_reward"], linewidth=2, color='#f39c12')
        ax7.set_xlabel("Epoch")
        ax7.set_ylabel("Intrinsic Reward")
        ax7.set_title("Intrinsic Reward", fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)

    # Episode length
    ax8 = fig.add_subplot(gs[2, 2])
    if log.get("episode_length"):
        ax8.plot(epochs, log["episode_length"], linewidth=2, color='#34495e')
        ax8.set_xlabel("Epoch")
        ax8.set_ylabel("Steps")
        ax8.set_title("Episode Length", fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)

    plt.suptitle("Training Summary Dashboard", fontsize=16, fontweight='bold', y=0.995)

    if output_dir:
        output_path = output_dir / "summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Internal Dimension AI experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to log directory containing training_log.json",
    )
    parser.add_argument(
        "--type",
        choices=[
            "all",
            "learning_curves",
            "internal_dims",
            "consciousness",
            "intrinsic",
            "exploration",
            "summary",
        ],
        default="all",
        help="Type of plot to generate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for plots (default: display only)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved plots (default: 300)",
    )

    args = parser.parse_args()

    # Update DPI
    plt.rcParams["savefig.dpi"] = args.dpi

    # Load log
    try:
        log = load_training_log(args.log)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    # Create output directory if needed
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving plots to: {args.output}")

    # Generate plots
    plot_funcs = {
        "learning_curves": plot_learning_curves,
        "internal_dims": plot_internal_dimensions,
        "consciousness": plot_consciousness_metrics,
        "intrinsic": plot_intrinsic_rewards,
        "exploration": plot_exploration,
        "summary": plot_summary,
    }

    if args.type == "all":
        for plot_func in plot_funcs.values():
            try:
                plot_func(log, args.output)
            except Exception as e:
                logger.error(f"Error generating plot: {e}")
    else:
        try:
            plot_funcs[args.type](log, args.output)
        except Exception as e:
            logger.error(f"Error generating plot: {e}")
            return 1

    logger.info("Visualization complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
