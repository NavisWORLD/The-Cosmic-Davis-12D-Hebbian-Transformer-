#!/usr/bin/env python3
"""
Baseline Comparison - Compare IDN vs Standard Network

Trains both Internal Dimension Network and standard baseline
on the same task, then compares performance and generates plots.

Run: python examples/02_baseline_comparison.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.core.network import InternalDimensionNetwork, BaselineNetwork
from src.environments.gridworld import GridWorld
from src.training.trainer import PPOTrainer
from src.evaluation.visualizations import InternalDimensionVisualizer

def main():
    print("="*70)
    print("BASELINE COMPARISON: IDN vs Standard Network")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = GridWorld(size=8, reward_goal=1.0, reward_step=-0.01, goal_position=(7,7))

    # Train IDN
    print("\n1. Training Internal Dimension Network...")
    idn_model = InternalDimensionNetwork(input_dim=2, hidden_dim=64, output_dim=4, device=device)
    idn_trainer = PPOTrainer(idn_model, env, device=device, use_tensorboard=False, use_wandb=False,
                            intrinsic_reward_weight=0.1, log_interval=20)
    idn_history = idn_trainer.train(num_episodes=200, steps_per_episode=200)

    # Train Baseline
    print("\n2. Training Baseline Network...")
    baseline_model = BaselineNetwork(input_dim=2, hidden_dim=64, output_dim=4, device=device)
    baseline_trainer = PPOTrainer(baseline_model, env, device=device, use_tensorboard=False, use_wandb=False, log_interval=20)
    baseline_history = baseline_trainer.train(num_episodes=200, steps_per_episode=200)

    # Compare
    print("\n3. Comparison Results:")
    print("-"*70)
    idn_final = np.mean(idn_history['episode_rewards'][-20:])
    baseline_final = np.mean(baseline_history['episode_rewards'][-20:])
    print(f"   IDN Final Reward:      {idn_final:.3f}")
    print(f"   Baseline Final Reward: {baseline_final:.3f}")
    print(f"   Improvement:           {(idn_final - baseline_final):.3f} ({((idn_final/baseline_final - 1)*100):.1f}%)")

    # Visualize
    viz = InternalDimensionVisualizer()
    output_dir = Path('outputs/baseline_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    viz.plot_baseline_comparison(idn_history, baseline_history,
                                 save_path=str(output_dir / 'comparison.png'), show=True)

    print(f"\n   Results saved to {output_dir}/")
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
