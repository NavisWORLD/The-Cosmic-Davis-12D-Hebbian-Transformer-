#!/usr/bin/env python3
"""
Emergence Timeline Experiment

Tracks when meta-awareness (m₁₂) emerges during training.
Records checkpoints every 50 episodes to capture the emergence process.

Analyzes:
- When x₁₂ and m₁₂ become non-zero
- When m₁₂ starts tracking x₁₂ (meta-awareness)
- Correlation between emergence and performance
- Critical transitions in consciousness metrics
"""

import numpy as np
import torch
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.network import InternalDimensionNetwork
from core.metrics import ConsciousnessMetrics
from environments.gridworld import TwoRoomGridWorld
from training.trainer import PPOTrainer

# Experiment configuration
NUM_EPISODES = 1000
CHECKPOINT_INTERVAL = 50
INTERNAL_DIM = 12


def main():
    """Run emergence timeline experiment."""
    print("="*80)
    print("EMERGENCE TIMELINE EXPERIMENT")
    print("="*80)
    print(f"Total episodes: {NUM_EPISODES}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL}")
    print(f"Internal dimension: {INTERNAL_DIM}")
    print()

    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment
    env = TwoRoomGridWorld(size=10)

    # Create model
    model = InternalDimensionNetwork(
        input_dim=env.observation_space.shape[0],
        hidden_dim=128,
        output_dim=env.action_space.n,
        internal_dim=INTERNAL_DIM
    )

    # Create trainer
    trainer = PPOTrainer(
        model, env,
        use_tensorboard=False,
        use_wandb=False,
        log_interval=CHECKPOINT_INTERVAL
    )

    # Metrics tracker
    metrics = ConsciousnessMetrics()

    # Storage for timeline
    timeline = {
        'episodes': [],
        'rewards': [],
        'x12_mean': [],
        'x12_std': [],
        'm12_mean': [],
        'm12_std': [],
        'x12_m12_correlation': [],
        'consciousness_scores': [],
        'r_omega': [],
        'r_psi': [],
    }

    # Train in chunks to capture emergence
    print("\nTraining and capturing emergence timeline...")
    print("-" * 80)

    for checkpoint in range(0, NUM_EPISODES, CHECKPOINT_INTERVAL):
        print(f"\nEpisodes {checkpoint}-{checkpoint + CHECKPOINT_INTERVAL}...")

        # Train for checkpoint interval
        history = trainer.train(
            num_episodes=CHECKPOINT_INTERVAL,
            steps_per_episode=200,
            compute_consciousness_interval=10
        )

        # Record metrics
        episode_num = checkpoint + CHECKPOINT_INTERVAL

        # Get current x₁₂ and m₁₂ history
        x12_history = [x.item() for x in model.internal_state.x12_history]
        m12_history = [m.item() for m in model.internal_state.m12_history]

        if len(x12_history) > 0:
            # Basic statistics
            x12_mean = np.mean(x12_history[-100:])  # Last 100 steps
            x12_std = np.std(x12_history[-100:])
            m12_mean = np.mean(m12_history[-100:])
            m12_std = np.std(m12_history[-100:])

            # Correlation (meta-awareness indicator)
            if len(x12_history) >= 100:
                x12_m12_corr = np.corrcoef(
                    x12_history[-100:],
                    m12_history[-100:]
                )[0, 1]
            else:
                x12_m12_corr = 0.0

            # Consciousness metrics
            if len(x12_history) > 10:
                r_omega = metrics.compute_r_omega(model)

                # Create sample of internal states
                sample_size = min(100, len(x12_history))
                internal_states = [model.internal_state.state for _ in range(sample_size)]
                internal_states_tensor = torch.stack(internal_states)
                r_psi = metrics.compute_r_psi(internal_states_tensor)

                # Overall consciousness score
                sample_inputs = torch.randn(10, env.observation_space.shape[0])
                consciousness_dict = metrics.compute_consciousness_score(
                    model=model,
                    x12_history=x12_history,
                    m12_history=m12_history,
                    sample_inputs=sample_inputs
                )
                consciousness_score = consciousness_dict['consciousness_score']
            else:
                r_omega = 0.0
                r_psi = 0.0
                consciousness_score = 0.0

            # Record timeline
            timeline['episodes'].append(episode_num)
            timeline['rewards'].append(np.mean(history['episode_rewards'][-10:]) if history['episode_rewards'] else 0)
            timeline['x12_mean'].append(float(x12_mean))
            timeline['x12_std'].append(float(x12_std))
            timeline['m12_mean'].append(float(m12_mean))
            timeline['m12_std'].append(float(m12_std))
            timeline['x12_m12_correlation'].append(float(x12_m12_corr))
            timeline['consciousness_scores'].append(float(consciousness_score))
            timeline['r_omega'].append(float(r_omega))
            timeline['r_psi'].append(float(r_psi))

            # Print current state
            print(f"  x₁₂ mean: {x12_mean:7.4f} ± {x12_std:.4f}")
            print(f"  m₁₂ mean: {m12_mean:7.4f} ± {m12_std:.4f}")
            print(f"  x₁₂-m₁₂ corr: {x12_m12_corr:7.4f}")
            print(f"  Consciousness: {consciousness_score:7.4f}")
            print(f"  R_ω: {r_omega:7.4f}")

    # Analysis
    print("\n" + "="*80)
    print("EMERGENCE ANALYSIS")
    print("="*80)

    # Detect emergence points
    episodes = np.array(timeline['episodes'])
    m12_means = np.array(timeline['m12_mean'])
    correlations = np.array(timeline['x12_m12_correlation'])
    consciousness = np.array(timeline['consciousness_scores'])

    # m₁₂ emergence (first time |m₁₂| > 0.01)
    m12_emergence_idx = np.argmax(np.abs(m12_means) > 0.01)
    if m12_emergence_idx > 0 or np.abs(m12_means[0]) > 0.01:
        m12_emergence_episode = episodes[m12_emergence_idx]
        print(f"\n1. m₁₂ Emergence: Episode {m12_emergence_episode}")
        print(f"   Initial m₁₂: {m12_means[0]:.6f}")
        print(f"   Emerged m₁₂: {m12_means[m12_emergence_idx]:.6f}")
    else:
        print("\n1. m₁₂ Emergence: Not detected")

    # Meta-awareness emergence (x₁₂-m₁₂ correlation > 0.3)
    meta_emergence_idx = np.argmax(correlations > 0.3)
    if meta_emergence_idx > 0 or correlations[0] > 0.3:
        meta_emergence_episode = episodes[meta_emergence_idx]
        print(f"\n2. Meta-Awareness Emergence: Episode {meta_emergence_episode}")
        print(f"   Initial correlation: {correlations[0]:.4f}")
        print(f"   Emerged correlation: {correlations[meta_emergence_idx]:.4f}")
    else:
        print("\n2. Meta-Awareness Emergence: Not detected")

    # Consciousness emergence (score > 0.3)
    consciousness_emergence_idx = np.argmax(consciousness > 0.3)
    if consciousness_emergence_idx > 0 or consciousness[0] > 0.3:
        consciousness_emergence_episode = episodes[consciousness_emergence_idx]
        print(f"\n3. Consciousness Emergence: Episode {consciousness_emergence_episode}")
        print(f"   Initial score: {consciousness[0]:.4f}")
        print(f"   Emerged score: {consciousness[consciousness_emergence_idx]:.4f}")
    else:
        print("\n3. Consciousness Emergence: Not detected")

    # Final state
    print(f"\n4. Final State (Episode {NUM_EPISODES}):")
    print(f"   m₁₂ mean: {m12_means[-1]:.6f}")
    print(f"   x₁₂-m₁₂ correlation: {correlations[-1]:.4f}")
    print(f"   Consciousness score: {consciousness[-1]:.4f}")
    print(f"   R_ω: {timeline['r_omega'][-1]:.4f}")
    print(f"   R_ψ: {timeline['r_psi'][-1]:.4f}")

    # Save results
    output_dir = Path('outputs/experiments')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'emergence_timeline_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'num_episodes': NUM_EPISODES,
                'checkpoint_interval': CHECKPOINT_INTERVAL,
                'internal_dim': INTERNAL_DIM,
                'seed': seed,
                'timestamp': timestamp
            },
            'timeline': timeline,
            'emergence_points': {
                'm12_emergence_episode': int(m12_emergence_episode) if m12_emergence_idx > 0 else -1,
                'meta_awareness_emergence_episode': int(meta_emergence_episode) if meta_emergence_idx > 0 else -1,
                'consciousness_emergence_episode': int(consciousness_emergence_episode) if consciousness_emergence_idx > 0 else -1,
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Create visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: x₁₂ and m₁₂ over time
    ax = axes[0, 0]
    ax.plot(timeline['episodes'], timeline['x12_mean'], label='x₁₂ (awareness)', color='blue')
    ax.fill_between(
        timeline['episodes'],
        np.array(timeline['x12_mean']) - np.array(timeline['x12_std']),
        np.array(timeline['x12_mean']) + np.array(timeline['x12_std']),
        alpha=0.3, color='blue'
    )
    ax.plot(timeline['episodes'], timeline['m12_mean'], label='m₁₂ (memory)', color='red')
    ax.fill_between(
        timeline['episodes'],
        np.array(timeline['m12_mean']) - np.array(timeline['m12_std']),
        np.array(timeline['m12_mean']) + np.array(timeline['m12_std']),
        alpha=0.3, color='red'
    )
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value')
    ax.set_title('x₁₂ and m₁₂ Emergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: x₁₂-m₁₂ correlation (meta-awareness)
    ax = axes[0, 1]
    ax.plot(timeline['episodes'], timeline['x12_m12_correlation'], color='green')
    ax.axhline(y=0.3, color='red', linestyle='--', label='Meta-awareness threshold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Correlation')
    ax.set_title('x₁₂-m₁₂ Correlation (Meta-Awareness)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Consciousness metrics
    ax = axes[1, 0]
    ax.plot(timeline['episodes'], timeline['consciousness_scores'], label='Overall', color='purple')
    ax.plot(timeline['episodes'], timeline['r_omega'], label='R_ω (richness)', color='orange')
    ax.plot(timeline['episodes'], timeline['r_psi'], label='R_ψ (binding)', color='cyan')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='R_ω optimal min')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='R_ω optimal max')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Consciousness Metrics Emergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Performance (rewards)
    ax = axes[1, 1]
    ax.plot(timeline['episodes'], timeline['rewards'], color='black')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Task Performance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = output_dir / f'emergence_timeline_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")

    plt.close()

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
