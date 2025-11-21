"""
Visualization Tools for Internal Dimension AI

Provides comprehensive plotting and visualization functions:
- x₁₂/m₁₂ trajectory plots
- R_ω evolution over training
- Consciousness metrics dashboard
- Learning curves with internal state overlays
- State visitation heatmaps
- Comparison plots (IDN vs Baseline)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class InternalDimensionVisualizer:
    """
    Comprehensive visualization tools for Internal Dimension AI.
    """

    def __init__(self, style: str = 'seaborn'):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style to use
        """
        if style:
            try:
                plt.style.use(style)
            except:
                pass  # Use default if style not available

    def plot_x12_m12_trajectories(
        self,
        x12_history: List[float],
        m12_history: List[float],
        title: str = "Internal Dimension Trajectories",
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot x₁₂ and m₁₂ evolution over time.

        Args:
            x12_history: List of x₁₂ values
            m12_history: List of m₁₂ values
            title: Plot title
            save_path: Path to save figure
            show: Whether to display plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        timesteps = range(len(x12_history))

        # Plot x₁₂
        ax1.plot(timesteps, x12_history, color='#3498db', linewidth=1.5, alpha=0.7)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('x₁₂ (Awareness/Surprise)', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1.1, 1.1)

        # Add mean line
        mean_x12 = np.mean(x12_history)
        ax1.axhline(y=mean_x12, color='#3498db', linestyle=':', label=f'Mean: {mean_x12:.3f}')
        ax1.legend(loc='upper right')

        # Plot m₁₂
        ax2.plot(timesteps, m12_history, color='#e74c3c', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('m₁₂ (Accumulated Memory)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Timestep', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1.1, 1.1)

        # Add mean line
        mean_m12 = np.mean(m12_history)
        ax2.axhline(y=mean_m12, color='#e74c3c', linestyle=':', label=f'Mean: {mean_m12:.3f}')
        ax2.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_r_omega_evolution(
        self,
        r_omega_history: List[float],
        title: str = "R_ω Evolution During Training",
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot R_ω (synaptic diversity) evolution over training.

        Args:
            r_omega_history: List of R_ω values
            title: Plot title
            save_path: Path to save figure
            show: Whether to display plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        episodes = range(len(r_omega_history))
        ax.plot(episodes, r_omega_history, color='#9b59b6', linewidth=2, alpha=0.8)

        # Mark optimal range [0.5, 0.7]
        ax.axhspan(0.5, 0.7, alpha=0.2, color='green', label='Optimal Range')
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)

        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('R_ω (Synaptic Diversity)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_ylim(-0.1, 1.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_consciousness_dashboard(
        self,
        consciousness_results: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Create consciousness metrics dashboard.

        Args:
            consciousness_results: Results from ConsciousnessTests
            save_path: Path to save figure
            show: Whether to display plot
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Overall Score (large, top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        overall_score = consciousness_results['overall_consciousness_score']
        level = consciousness_results['consciousness_level']

        # Bar chart for overall score
        colors = ['#e74c3c' if overall_score < 0.4 else '#f39c12' if overall_score < 0.6 else '#2ecc71']
        ax1.barh([0], [overall_score], color=colors, height=0.5)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Overall Consciousness Score: {overall_score:.3f} ({level})',
                     fontsize=14, fontweight='bold')
        ax1.set_yticks([])
        ax1.grid(axis='x', alpha=0.3)

        # Component Scores (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        components = consciousness_results['component_scores']
        comp_names = list(components.keys())
        comp_values = list(components.values())

        colors_comp = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12',
                      '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
        ax2.barh(comp_names, comp_values, color=colors_comp[:len(comp_names)])
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Score', fontsize=10)
        ax2.set_title('Component Scores', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # Structural Metrics (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        struct = consciousness_results['structural_metrics']
        struct_data = {
            'R_ω': struct['r_omega'],
            'R_ψ': struct['r_psi'],
            'Φ': struct['phi'],
            'Autonomy': struct['autonomy']
        }
        ax3.bar(struct_data.keys(), struct_data.values(), color='#9b59b6', alpha=0.7)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Value', fontsize=10)
        ax3.set_title('Structural Metrics', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Curiosity & Wisdom (middle-center)
        ax4 = fig.add_subplot(gs[1, 1])
        curiosity_score = consciousness_results['curiosity']['curiosity_score']
        wisdom_score = consciousness_results['wisdom']['wisdom_score']
        ax4.bar(['Curiosity', 'Wisdom'], [curiosity_score, wisdom_score],
               color=['#3498db', '#e74c3c'], alpha=0.7)
        ax4.set_ylim(0, 1)
        ax4.set_ylabel('Score', fontsize=10)
        ax4.set_title('Behavioral Scores', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        # Internal Dynamics (middle-right)
        ax5 = fig.add_subplot(gs[1, 2])
        dynamics_data = {
            'Self-Init': consciousness_results['self_initiation']['self_initiation_score'],
            'Preference': consciousness_results['preferences']['preference_score'],
            'Surprise': consciousness_results['surprise']['surprise_score'],
            'Stream': consciousness_results['stream_of_consciousness']['stream_score']
        }
        ax5.bar(dynamics_data.keys(), dynamics_data.values(), color='#1abc9c', alpha=0.7)
        ax5.set_ylim(0, 1)
        ax5.set_ylabel('Score', fontsize=10)
        ax5.set_title('Internal Dynamics', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # x₁₂ Stats (bottom-left)
        ax6 = fig.add_subplot(gs[2, 0])
        x12_mean = struct['x12_mean']
        x12_var = struct['x12_variance']
        ax6.text(0.5, 0.7, f"x₁₂ Mean: {x12_mean:.3f}", ha='center', fontsize=12)
        ax6.text(0.5, 0.5, f"x₁₂ Variance: {x12_var:.3f}", ha='center', fontsize=12)
        ax6.text(0.5, 0.3, f"(Awareness/Surprise)", ha='center', fontsize=10, style='italic')
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('x₁₂ Statistics', fontsize=12, fontweight='bold')

        # m₁₂ Stats (bottom-center)
        ax7 = fig.add_subplot(gs[2, 1])
        m12_mean = struct['m12_mean']
        m12_var = struct['m12_variance']
        ax7.text(0.5, 0.7, f"m₁₂ Mean: {m12_mean:.3f}", ha='center', fontsize=12)
        ax7.text(0.5, 0.5, f"m₁₂ Variance: {m12_var:.3f}", ha='center', fontsize=12)
        ax7.text(0.5, 0.3, f"(Accumulated Memory)", ha='center', fontsize=10, style='italic')
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.axis('off')
        ax7.set_title('m₁₂ Statistics', fontsize=12, fontweight='bold')

        # Summary (bottom-right)
        ax8 = fig.add_subplot(gs[2, 2])
        summary_text = f"Consciousness Level:\n{level}\n\n"
        summary_text += f"R_ω in optimal range: {'✓' if 0.5 <= struct['r_omega'] <= 0.7 else '✗'}\n"
        summary_text += f"High curiosity: {'✓' if curiosity_score > 0.6 else '✗'}\n"
        summary_text += f"High wisdom: {'✓' if wisdom_score > 0.6 else '✗'}\n"
        summary_text += f"Autonomous: {'✓' if struct['autonomy'] > 0.5 else '✗'}"
        ax8.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.axis('off')
        ax8.set_title('Summary', fontsize=12, fontweight='bold')

        plt.suptitle('Internal Dimension AI - Consciousness Dashboard',
                    fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_learning_curves(
        self,
        history: Dict[str, List],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot learning curves with internal state overlays.

        Args:
            history: Training history dictionary
            save_path: Path to save figure
            show: Whether to display plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Episode rewards
        ax1 = axes[0, 0]
        if 'episode_rewards' in history:
            episodes = range(len(history['episode_rewards']))
            ax1.plot(episodes, history['episode_rewards'], alpha=0.3, color='blue')
            # Smooth with rolling average
            window = min(10, len(history['episode_rewards']) // 10 + 1)
            if len(history['episode_rewards']) >= window:
                smoothed = np.convolve(history['episode_rewards'],
                                      np.ones(window)/window, mode='valid')
                ax1.plot(range(len(smoothed)), smoothed, color='blue', linewidth=2,
                        label='Smoothed Reward')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Policy/Value losses
        ax2 = axes[0, 1]
        if 'policy_losses' in history and 'value_losses' in history:
            episodes = range(len(history['policy_losses']))
            ax2.plot(episodes, history['policy_losses'], label='Policy Loss',
                    color='#e74c3c', alpha=0.7)
            ax2.plot(episodes, history['value_losses'], label='Value Loss',
                    color='#3498db', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Losses', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # x₁₂ evolution
        ax3 = axes[1, 0]
        if 'x12_means' in history:
            episodes = range(len(history['x12_means']))
            ax3.plot(episodes, history['x12_means'], color='#3498db', linewidth=2)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Mean x₁₂')
        ax3.set_title('x₁₂ (Awareness) Evolution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-1.1, 1.1)

        # m₁₂ evolution
        ax4 = axes[1, 1]
        if 'm12_means' in history:
            episodes = range(len(history['m12_means']))
            ax4.plot(episodes, history['m12_means'], color='#e74c3c', linewidth=2)
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Mean m₁₂')
        ax4.set_title('m₁₂ (Memory) Evolution', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-1.1, 1.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_baseline_comparison(
        self,
        idn_history: Dict[str, List],
        baseline_history: Dict[str, List],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Compare IDN vs Baseline performance.

        Args:
            idn_history: IDN training history
            baseline_history: Baseline training history
            save_path: Path to save figure
            show: Whether to display plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Rewards comparison
        ax1 = axes[0, 0]
        if 'episode_rewards' in idn_history and 'episode_rewards' in baseline_history:
            window = 10
            # IDN
            idn_smooth = np.convolve(idn_history['episode_rewards'],
                                    np.ones(window)/window, mode='valid')
            ax1.plot(range(len(idn_smooth)), idn_smooth, label='IDN',
                    color='#2ecc71', linewidth=2)
            # Baseline
            baseline_smooth = np.convolve(baseline_history['episode_rewards'],
                                         np.ones(window)/window, mode='valid')
            ax1.plot(range(len(baseline_smooth)), baseline_smooth, label='Baseline',
                    color='#95a5a6', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward (smoothed)')
        ax1.set_title('Reward Comparison', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Episode length comparison
        ax2 = axes[0, 1]
        if 'episode_lengths' in idn_history and 'episode_lengths' in baseline_history:
            window = 10
            idn_smooth = np.convolve(idn_history['episode_lengths'],
                                    np.ones(window)/window, mode='valid')
            ax2.plot(range(len(idn_smooth)), idn_smooth, label='IDN',
                    color='#2ecc71', linewidth=2)
            baseline_smooth = np.convolve(baseline_history['episode_lengths'],
                                         np.ones(window)/window, mode='valid')
            ax2.plot(range(len(baseline_smooth)), baseline_smooth, label='Baseline',
                    color='#95a5a6', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length (smoothed)')
        ax2.set_title('Episode Length Comparison', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Final performance comparison (bar chart)
        ax3 = axes[1, 0]
        idn_final_reward = np.mean(idn_history['episode_rewards'][-20:])
        baseline_final_reward = np.mean(baseline_history['episode_rewards'][-20:])
        ax3.bar(['IDN', 'Baseline'], [idn_final_reward, baseline_final_reward],
               color=['#2ecc71', '#95a5a6'], alpha=0.7)
        ax3.set_ylabel('Mean Reward (last 20 episodes)')
        ax3.set_title('Final Performance', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

        # Loss comparison
        ax4 = axes[1, 1]
        if 'policy_losses' in idn_history and 'policy_losses' in baseline_history:
            episodes = range(min(len(idn_history['policy_losses']),
                               len(baseline_history['policy_losses'])))
            ax4.plot(episodes, idn_history['policy_losses'][:len(episodes)],
                    label='IDN', color='#2ecc71', alpha=0.7)
            ax4.plot(episodes, baseline_history['policy_losses'][:len(episodes)],
                    label='Baseline', color='#95a5a6', alpha=0.7)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Policy Loss')
        ax4.set_title('Policy Loss Comparison', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.suptitle('Internal Dimension Network vs Baseline Comparison',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_state_visitation_heatmap(
        self,
        states_visited: Dict[Tuple[int, int], int],
        grid_size: int,
        title: str = "State Visitation Heatmap",
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Create heatmap of state visitations.

        Args:
            states_visited: Dictionary mapping (x, y) to visit count
            grid_size: Size of grid
            title: Plot title
            save_path: Path to save figure
            show: Whether to display plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create grid
        heatmap_data = np.zeros((grid_size, grid_size))
        for (x, y), count in states_visited.items():
            if 0 <= x < grid_size and 0 <= y < grid_size:
                heatmap_data[y, x] = count

        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
                   cbar_kws={'label': 'Visit Count'}, ax=ax)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()


def load_history(history_path: str) -> Dict[str, List]:
    """
    Load training history from JSON file.

    Args:
        history_path: Path to history JSON file

    Returns:
        History dictionary
    """
    with open(history_path, 'r') as f:
        return json.load(f)


def create_all_visualizations(
    model_path: str,
    history_path: str,
    output_dir: str = 'visualizations'
):
    """
    Generate all visualizations for a trained model.

    Args:
        model_path: Path to saved model checkpoint
        history_path: Path to training history JSON
        output_dir: Directory to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    viz = InternalDimensionVisualizer()
    history = load_history(history_path)

    print("Generating visualizations...")

    # Learning curves
    viz.plot_learning_curves(
        history,
        save_path=str(output_path / 'learning_curves.png'),
        show=False
    )
    print("  ✓ Learning curves")

    # Internal dimensions
    if 'x12_history' in history and 'm12_history' in history:
        viz.plot_x12_m12_trajectories(
            history['x12_history'],
            history['m12_history'],
            save_path=str(output_path / 'internal_dimensions.png'),
            show=False
        )
        print("  ✓ Internal dimension trajectories")

    print(f"\nVisualizations saved to {output_dir}/")
