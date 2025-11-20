"""
Visualization Utilities for 12D Cosmic Synapse Transformer

Functions for plotting x12 evolution, Hebbian connectivity, attention patterns,
and training metrics.

Author: Cory Shane Davis
License: MIT
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import torch


def plot_x12_evolution(
    x12_history: List[torch.Tensor],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot x12 internal state evolution over time.

    Args:
        x12_history: List of x12 tensors from different training steps
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    # Convert to numpy
    x12_array = np.array([x.cpu().detach().numpy() for x in x12_history])

    n_layers = x12_array.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: All layers over time
    for layer in range(n_layers):
        axes[0].plot(x12_array[:, layer], label=f'Layer {layer}', alpha=0.7)

    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('x12 Value')
    axes[0].set_title('x12 Evolution Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Plot 2: Final distribution
    final_x12 = x12_array[-1]
    axes[1].bar(range(n_layers), final_x12)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Final x12 Value')
    axes[1].set_title('Final x12 Distribution')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_hebbian_matrix(
    omega_matrix: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> None:
    """
    Plot Hebbian connectivity matrix as heatmap.

    Args:
        omega_matrix: Hebbian connectivity matrix [n_heads, seq_len, seq_len]
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    omega_np = omega_matrix.cpu().detach().numpy()

    # Average over heads
    avg_omega = omega_np.mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(avg_omega, cmap='RdBu_r', aspect='auto')
    ax.set_xlabel('Position')
    ax.set_ylabel('Position')
    ax.set_title('Hebbian Connectivity Matrix (Ω)\nAveraged over attention heads')

    plt.colorbar(im, ax=ax, label='Connection Strength')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_attention_patterns(
    attention_weights: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 4)
) -> None:
    """
    Visualize attention patterns.

    Args:
        attention_weights: Attention weights [n_heads, seq_len, seq_len]
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    attn_np = attention_weights.cpu().detach().numpy()
    n_heads = attn_np.shape[0]

    # Plot first 4 heads
    n_plot = min(4, n_heads)

    fig, axes = plt.subplots(1, n_plot, figsize=figsize)
    if n_plot == 1:
        axes = [axes]

    for i in range(n_plot):
        im = axes[i].imshow(attn_np[i], cmap='viridis', aspect='auto')
        axes[i].set_title(f'Head {i}')
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_training_curves(
    metrics: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10)
) -> None:
    """
    Plot comprehensive training metrics.

    Args:
        metrics: Dictionary with keys like 'train_loss', 'val_loss', 'x12_mean', 'lr'
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Loss curves
    if 'train_loss' in metrics:
        axes[0, 0].plot(metrics['train_loss'], label='Train', alpha=0.7)
    if 'val_loss' in metrics:
        axes[0, 0].plot(metrics['val_loss'], label='Validation', alpha=0.7)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # x12 evolution
    if 'x12_mean' in metrics:
        axes[0, 1].plot(metrics['x12_mean'], color='purple')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Mean |x12|')
        axes[0, 1].set_title('Internal State Magnitude')
        axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    if 'lr' in metrics:
        axes[1, 0].plot(metrics['lr'], color='orange')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)

    # Gradient norm
    if 'grad_norm' in metrics:
        axes[1, 1].plot(metrics['grad_norm'], color='green', alpha=0.5)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_title('Gradient Magnitude')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_lorenz_attractor(
    lorenz_history: List[torch.Tensor],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10)
) -> None:
    """
    Plot Lorenz attractor trajectory in 3D.

    Args:
        lorenz_history: List of Lorenz state tensors [3]
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Convert to numpy
    lorenz_array = np.array([l.cpu().detach().numpy() for l in lorenz_history])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(lorenz_array[:, 0], lorenz_array[:, 1], lorenz_array[:, 2],
            lw=0.5, alpha=0.7)

    # Plot start and end points
    ax.scatter([lorenz_array[0, 0]], [lorenz_array[0, 1]], [lorenz_array[0, 2]],
               c='green', marker='o', s=100, label='Start')
    ax.scatter([lorenz_array[-1, 0]], [lorenz_array[-1, 1]], [lorenz_array[-1, 2]],
               c='red', marker='o', s=100, label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Lorenz Attractor Trajectory')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def create_dashboard(
    metrics: dict,
    x12_history: Optional[List[torch.Tensor]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive training dashboard.

    Args:
        metrics: Training metrics dictionary
        x12_history: Optional x12 evolution history
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main loss plot (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if 'train_loss' in metrics:
        ax1.plot(metrics['train_loss'], label='Train', alpha=0.7)
    if 'val_loss' in metrics:
        ax1.plot(metrics['val_loss'], label='Validation', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Learning rate
    ax2 = fig.add_subplot(gs[0, 2])
    if 'lr' in metrics:
        ax2.plot(metrics['lr'], color='orange')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('LR')
    ax2.set_title('Learning Rate')
    ax2.grid(True, alpha=0.3)

    # x12 evolution
    if x12_history and len(x12_history) > 0:
        ax3 = fig.add_subplot(gs[1, :])
        x12_array = np.array([x.cpu().detach().numpy() for x in x12_history])
        for layer in range(x12_array.shape[1]):
            ax3.plot(x12_array[:, layer], label=f'Layer {layer}', alpha=0.7)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('x12')
        ax3.set_title('Internal State Evolution (x12)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Statistics
    ax4 = fig.add_subplot(gs[2, 0])
    if 'train_loss' in metrics and len(metrics['train_loss']) > 0:
        stats_text = f"Final Train Loss: {metrics['train_loss'][-1]:.4f}\n"
        if 'val_loss' in metrics and len(metrics['val_loss']) > 0:
            stats_text += f"Final Val Loss: {metrics['val_loss'][-1]:.4f}\n"
        stats_text += f"Total Iterations: {len(metrics['train_loss'])}\n"
        if x12_history:
            final_x12 = x12_history[-1].abs().mean().item()
            stats_text += f"Final |x12|: {final_x12:.4f}"

        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                 family='monospace')
        ax4.axis('off')
        ax4.set_title('Summary Statistics')

    # Gradient norm
    ax5 = fig.add_subplot(gs[2, 1])
    if 'grad_norm' in metrics:
        ax5.plot(metrics['grad_norm'], color='green', alpha=0.5)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Gradient Norm')
        ax5.set_title('Gradient Magnitude')
        ax5.grid(True, alpha=0.3)

    # Performance metrics
    ax6 = fig.add_subplot(gs[2, 2])
    if 'tokens_per_sec' in metrics:
        ax6.plot(metrics['tokens_per_sec'], color='blue')
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Tokens/sec')
        ax6.set_title('Training Speed')
        ax6.grid(True, alpha=0.3)

    fig.suptitle('12D Cosmic Synapse Transformer - Training Dashboard',
                 fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved dashboard to {save_path}")
    else:
        plt.show()
