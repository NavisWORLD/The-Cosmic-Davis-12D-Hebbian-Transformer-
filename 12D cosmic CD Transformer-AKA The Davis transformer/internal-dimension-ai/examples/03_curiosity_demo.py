#!/usr/bin/env python3
"""
Curiosity Demo - Demonstrate curiosity-driven exploration

Shows how Internal Dimension Network explores novel environments
driven by x₁₂ (awareness/surprise).

Run: python examples/03_curiosity_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.core.network import InternalDimensionNetwork
from src.environments.gridworld import TwoRoomGridWorld
from src.training.trainer import PPOTrainer
from src.evaluation.curiosity_tests import CuriosityTests

def main():
    print("="*70)
    print("CURIOSITY DEMO - Exploring Novel Environments")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create two-room environment
    print("\n1. Creating Two-Room Environment...")
    print("   Room A: Familiar (no rewards)")
    print("   Room B: Novel (contains features)")
    env = TwoRoomGridWorld(room_size=5, novel_features=True)

    # Train IDN with high curiosity weight
    print("\n2. Training Curiosity-Driven Agent...")
    model = InternalDimensionNetwork(input_dim=2, hidden_dim=64, output_dim=4, device=device)
    trainer = PPOTrainer(model, env, device=device, use_tensorboard=False, use_wandb=False,
                        intrinsic_reward_weight=0.5,  # High curiosity bonus
                        intrinsic_reward_method='curiosity', log_interval=10)

    history = trainer.train(num_episodes=50, steps_per_episode=100)

    # Test curiosity
    print("\n3. Running Curiosity Tests...")
    print("-"*70)
    curiosity_tests = CuriosityTests(device=device)

    # Novel room exploration test
    test1 = curiosity_tests.test_novel_room_exploration(model, num_episodes=10)
    print(f"   Novel Room Exploration:  {test1['proportion_in_novel_room']:.1%}")
    print(f"   x₁₂ boost in novel room: {test1['x12_novelty_boost']:.3f}")
    print(f"   Test Passed:             {'✓' if test1['test_passed'] else '✗'}")

    # Overall curiosity score
    results = curiosity_tests.compute_curiosity_score(model)
    print(f"\n   Overall Curiosity Score: {results['curiosity_score']:.3f}")
    print(f"   Curiosity Level:         {results['curiosity_level']}")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
