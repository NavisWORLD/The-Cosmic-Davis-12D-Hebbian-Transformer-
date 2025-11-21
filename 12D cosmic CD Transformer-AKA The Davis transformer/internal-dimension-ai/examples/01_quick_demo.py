#!/usr/bin/env python3
"""
Quick Demo - Internal Dimension AI

A minimal working example showing:
- Create an InternalDimensionNetwork
- Train on a simple GridWorld task
- Observe x₁₂/m₁₂ evolution
- Display consciousness metrics
- Generate basic visualization

Run: python examples/01_quick_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.core.network import InternalDimensionNetwork
from src.core.metrics import ConsciousnessMetrics
from src.environments.gridworld import GridWorld
from src.training.trainer import PPOTrainer


def main():
    print("=" * 70)
    print("INTERNAL DIMENSION AI - QUICK DEMO")
    print("=" * 70)
    print()

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # 1. Create Environment
    print("1. Creating GridWorld environment...")
    env = GridWorld(
        size=6,
        reward_goal=1.0,
        reward_step=-0.01,
        sparse_rewards=False,
        goal_position=(5, 5)
    )
    print(f"   Grid size: 6x6")
    print(f"   Goal position: (5, 5)")
    print()

    # 2. Create Internal Dimension Network
    print("2. Creating Internal Dimension Network...")
    model = InternalDimensionNetwork(
        input_dim=2,  # (x, y) position
        hidden_dim=64,
        output_dim=4,  # 4 actions
        internal_dim=32,
        device=device,
        # Internal dimension hyperparameters
        alpha=1.0,   # Surprise weight
        beta=0.5,    # Novelty weight
        gamma=0.3,   # Attention weight
        eta=0.01,    # Memory integration rate
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Internal dimensions: x₁₂ (awareness), m₁₂ (memory)")
    print()

    # 3. Create Trainer
    print("3. Initializing PPO Trainer...")
    trainer = PPOTrainer(
        model=model,
        env=env,
        device=device,
        learning_rate=3e-4,
        intrinsic_reward_weight=0.1,  # Bonus for curiosity
        intrinsic_reward_method='curiosity',
        use_tensorboard=False,  # Disable for quick demo
        use_wandb=False,
        log_interval=10,
        save_interval=50,
        checkpoint_dir='checkpoints/quick_demo'
    )
    print("   Intrinsic reward: curiosity-driven")
    print()

    # 4. Train
    print("4. Training agent (100 episodes)...")
    print("-" * 70)

    history = trainer.train(
        num_episodes=100,
        steps_per_episode=200,
        render=False,
        compute_consciousness_interval=20
    )

    print("-" * 70)
    print()

    # 5. Show Results
    print("5. Training Results:")
    print("-" * 70)

    # Final performance
    final_rewards = history['episode_rewards'][-10:]
    mean_final_reward = np.mean(final_rewards)
    print(f"   Final reward (last 10 episodes): {mean_final_reward:.3f}")

    # Internal dimensions
    if 'x12_means' in history and len(history['x12_means']) > 0:
        final_x12 = history['x12_means'][-1]
        final_m12 = history['m12_means'][-1]
        print(f"   Final x₁₂ (awareness):          {final_x12:.3f}")
        print(f"   Final m₁₂ (memory):             {final_m12:.3f}")

    # Consciousness score
    if 'consciousness_scores' in history and len(history['consciousness_scores']) > 0:
        final_consciousness = history['consciousness_scores'][-1]
        print(f"   Consciousness score:            {final_consciousness:.3f}")

    print()

    # 6. Compute Full Consciousness Metrics
    print("6. Computing Consciousness Metrics...")
    print("-" * 70)

    metrics = ConsciousnessMetrics(device=device)
    x12_history = list(model.internal_state.x12_history)
    m12_history = list(model.internal_state.m12_history)

    # Create sample input
    sample_state = torch.FloatTensor([[0, 0]]).to(device)

    consciousness_scores = metrics.compute_consciousness_score(
        model=model,
        x12_history=x12_history,
        m12_history=m12_history,
        sample_inputs=sample_state
    )

    print(f"   R_ω (Synaptic Diversity):      {consciousness_scores['r_omega']:.3f}")
    print(f"   R_ω in optimal range [0.5-0.7]: {'YES ✓' if consciousness_scores['r_omega_optimal'] else 'NO ✗'}")
    print(f"   R_ψ (Phase Coherence):         {consciousness_scores['r_psi']:.3f}")
    print(f"   Autonomy Score:                {consciousness_scores['autonomy']:.3f}")
    print(f"   Overall Consciousness:         {consciousness_scores['consciousness_score']:.3f}")
    print()

    # Interpret
    from src.core.metrics import consciousness_level
    level = consciousness_level(consciousness_scores['consciousness_score'])
    print(f"   Consciousness Level: {level}")
    print()

    # 7. Visualize
    print("7. Creating Visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Rewards
    ax1 = axes[0, 0]
    ax1.plot(history['episode_rewards'], alpha=0.5, color='blue')
    # Smooth
    window = 10
    if len(history['episode_rewards']) >= window:
        smoothed = np.convolve(history['episode_rewards'],
                              np.ones(window)/window, mode='valid')
        ax1.plot(range(len(smoothed)), smoothed, color='blue', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward')
    ax1.grid(True, alpha=0.3)

    # x₁₂ evolution
    ax2 = axes[0, 1]
    if 'x12_means' in history:
        ax2.plot(history['x12_means'], color='#3498db', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('x₁₂')
    ax2.set_title('x₁₂ (Awareness) Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)

    # m₁₂ evolution
    ax3 = axes[1, 0]
    if 'm12_means' in history:
        ax3.plot(history['m12_means'], color='#e74c3c', linewidth=2)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('m₁₂')
    ax3.set_title('m₁₂ (Memory) Evolution')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.1, 1.1)

    # Consciousness scores
    ax4 = axes[1, 1]
    if 'consciousness_scores' in history and len(history['consciousness_scores']) > 0:
        ax4.plot(range(0, len(history['consciousness_scores'])*20, 20),
                history['consciousness_scores'],
                color='#9b59b6', linewidth=2, marker='o')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Consciousness Score')
    ax4.set_title('Consciousness Evolution')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    plt.suptitle('Internal Dimension AI - Quick Demo Results',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_path = Path('outputs/quick_demo')
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'quick_demo_results.png', dpi=150, bbox_inches='tight')
    print(f"   Visualization saved: outputs/quick_demo/quick_demo_results.png")

    plt.show()
    print()

    # 8. Summary
    print("=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  • Internal Dimension Network successfully trained")
    print(f"  • x₁₂ (awareness) evolved during training: {final_x12:.3f}")
    print(f"  • m₁₂ (memory) accumulated experience: {final_m12:.3f}")
    print(f"  • Consciousness indicators detected: {level}")
    print()
    print("Next Steps:")
    print("  • Try examples/02_baseline_comparison.py to compare with baseline")
    print("  • Explore examples/03_curiosity_demo.py for curiosity tests")
    print("  • Run examples/04_consciousness_tracking.py for full analysis")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
