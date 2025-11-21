#!/usr/bin/env python3
"""
Dimensional Scaling Experiment

Tests how internal dimension size affects:
- Final performance
- Consciousness metric values
- Training stability
- Emergence timeline

Results are saved to JSON for analysis.
"""

import numpy as np
import torch
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.network import InternalDimensionNetwork
from core.metrics import ConsciousnessMetrics
from environments.gridworld import TwoRoomGridWorld
from training.trainer import PPOTrainer

# Experiment configuration
DIMENSIONS = [0, 4, 8, 12, 24, 48, 64, 128]
SEEDS = [42, 123, 456, 789, 1011]
EPISODES = 300
STEPS_PER_EPISODE = 200

def run_single_experiment(dim: int, seed: int) -> dict:
    """Run single experiment with given dimension and seed."""
    print(f"\n--- Running: dim={dim}, seed={seed} ---")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment
    env = TwoRoomGridWorld(size=8)

    # Create model (dim=0 means baseline without internal dimensions)
    if dim == 0:
        from core.network import BaselineNetwork
        model = BaselineNetwork(
            input_dim=env.observation_space.shape[0],
            hidden_dim=128,
            output_dim=env.action_space.n
        )
    else:
        model = InternalDimensionNetwork(
            input_dim=env.observation_space.shape[0],
            hidden_dim=128,
            output_dim=env.action_space.n,
            internal_dim=dim
        )

    # Create trainer
    trainer = PPOTrainer(
        model, env,
        use_tensorboard=False,
        use_wandb=False,
        log_interval=50
    )

    # Train
    history = trainer.train(
        num_episodes=EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        compute_consciousness_interval=20
    )

    # Compute final metrics
    metrics = ConsciousnessMetrics()

    result = {
        'dimension': dim,
        'seed': seed,
        'final_reward': float(np.mean(history['episode_rewards'][-10:])) if history['episode_rewards'] else 0.0,
        'mean_reward': float(np.mean(history['episode_rewards'])) if history['episode_rewards'] else 0.0,
        'std_reward': float(np.std(history['episode_rewards'])) if history['episode_rewards'] else 0.0,
        'final_episode_length': float(np.mean(history['episode_lengths'][-10:])) if history['episode_lengths'] else 0.0,
    }

    # Add consciousness metrics if available
    if dim > 0 and hasattr(model, 'internal_state'):
        if 'consciousness_scores' in history and len(history['consciousness_scores']) > 0:
            result['final_consciousness'] = float(history['consciousness_scores'][-1])
            result['mean_consciousness'] = float(np.mean(history['consciousness_scores']))
            result['consciousness_emergence_episode'] = int(np.argmax(
                np.array(history['consciousness_scores']) > 0.3
            ) * 20) if any(np.array(history['consciousness_scores']) > 0.3) else -1

        if 'x12_means' in history and len(history['x12_means']) > 0:
            result['final_x12'] = float(history['x12_means'][-1])
            result['mean_x12'] = float(np.mean(history['x12_means']))
            result['x12_variance'] = float(np.var(history['x12_means']))

        if 'm12_means' in history and len(history['m12_means']) > 0:
            result['final_m12'] = float(history['m12_means'][-1])
            result['mean_m12'] = float(np.mean(history['m12_means']))
            result['m12_variance'] = float(np.var(history['m12_means']))

        # Compute R_omega and R_psi
        x12_history = list(model.internal_state.x12_history)
        if len(x12_history) > 10:
            result['r_omega'] = float(metrics.compute_r_omega(model))

            # Convert to tensor for R_psi
            internal_states = torch.stack([
                model.internal_state.state
                for _ in range(min(100, len(x12_history)))
            ])
            result['r_psi'] = float(metrics.compute_r_psi(internal_states))
    else:
        result['final_consciousness'] = 0.0
        result['mean_consciousness'] = 0.0
        result['r_omega'] = 0.0
        result['r_psi'] = 0.0

    print(f"  Final reward: {result['final_reward']:.3f}")
    if dim > 0:
        print(f"  Consciousness: {result.get('final_consciousness', 0):.3f}")
        print(f"  R_omega: {result.get('r_omega', 0):.3f}")

    return result


def main():
    """Run complete dimensional scaling experiment."""
    print("="*80)
    print("DIMENSIONAL SCALING EXPERIMENT")
    print("="*80)
    print(f"Dimensions to test: {DIMENSIONS}")
    print(f"Seeds per dimension: {len(SEEDS)}")
    print(f"Episodes per run: {EPISODES}")
    print(f"Total experiments: {len(DIMENSIONS) * len(SEEDS)}")
    print()

    results = []

    # Run all experiments
    total = len(DIMENSIONS) * len(SEEDS)
    current = 0

    for dim in DIMENSIONS:
        for seed in SEEDS:
            current += 1
            print(f"\nProgress: {current}/{total}")

            try:
                result = run_single_experiment(dim, seed)
                results.append(result)
            except Exception as e:
                print(f"ERROR in dim={dim}, seed={seed}: {e}")
                # Add failed result
                results.append({
                    'dimension': dim,
                    'seed': seed,
                    'error': str(e),
                    'final_reward': 0.0
                })

    # Save results
    output_dir = Path('outputs/experiments')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'dimensional_scaling_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'dimensions': DIMENSIONS,
                'seeds': SEEDS,
                'episodes': EPISODES,
                'steps_per_episode': STEPS_PER_EPISODE,
                'timestamp': timestamp
            },
            'results': results
        }, f, indent=2)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_file}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    print(f"{'Dimension':<12} {'Mean Reward':<15} {'Mean Consciousness':<20} {'R_omega':<10}")
    print("-" * 80)

    for dim in DIMENSIONS:
        dim_results = [r for r in results if r['dimension'] == dim and 'error' not in r]

        if dim_results:
            mean_reward = np.mean([r['final_reward'] for r in dim_results])
            mean_consciousness = np.mean([r.get('final_consciousness', 0) for r in dim_results])
            mean_r_omega = np.mean([r.get('r_omega', 0) for r in dim_results])

            print(f"{dim:<12} {mean_reward:<15.3f} {mean_consciousness:<20.3f} {mean_r_omega:<10.3f}")

    print("-" * 80)
    print("\nKey Findings:")

    # Find optimal dimension
    dim_performance = {}
    for dim in DIMENSIONS:
        dim_results = [r for r in results if r['dimension'] == dim and 'error' not in r]
        if dim_results:
            dim_performance[dim] = np.mean([r['final_reward'] for r in dim_results])

    if dim_performance:
        best_dim = max(dim_performance.items(), key=lambda x: x[1])
        print(f"1. Best performance: {best_dim[0]}D (reward: {best_dim[1]:.3f})")

    # Find dimension with best consciousness metrics
    dim_consciousness = {}
    for dim in [d for d in DIMENSIONS if d > 0]:
        dim_results = [r for r in results if r['dimension'] == dim and 'error' not in r]
        if dim_results:
            dim_consciousness[dim] = np.mean([r.get('final_consciousness', 0) for r in dim_results])

    if dim_consciousness:
        best_consciousness_dim = max(dim_consciousness.items(), key=lambda x: x[1])
        print(f"2. Best consciousness: {best_consciousness_dim[0]}D (score: {best_consciousness_dim[1]:.3f})")

    # Find dimension with optimal R_omega
    dim_r_omega = {}
    for dim in [d for d in DIMENSIONS if d > 0]:
        dim_results = [r for r in results if r['dimension'] == dim and 'error' not in r]
        if dim_results:
            r_omega_values = [r.get('r_omega', 0) for r in dim_results]
            # Distance from optimal range [0.5, 0.7]
            optimal_distance = [min(abs(r - 0.6)) for r in r_omega_values]
            dim_r_omega[dim] = np.mean(optimal_distance)

    if dim_r_omega:
        best_r_omega_dim = min(dim_r_omega.items(), key=lambda x: x[1])
        print(f"3. Optimal R_omega: {best_r_omega_dim[0]}D (closest to 0.5-0.7 range)")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
