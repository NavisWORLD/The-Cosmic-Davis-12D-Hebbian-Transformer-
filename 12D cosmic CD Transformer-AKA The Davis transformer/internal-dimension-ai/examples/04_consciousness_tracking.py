#!/usr/bin/env python3
"""
Consciousness Tracking - Full consciousness analysis

Trains an agent while tracking all consciousness metrics,
then generates comprehensive consciousness report.

Run: python examples/04_consciousness_tracking.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.core.network import InternalDimensionNetwork
from src.environments.gridworld import GridWorld
from src.training.trainer import PPOTrainer
from src.evaluation.consciousness_tests import ConsciousnessTests
from src.evaluation.visualizations import InternalDimensionVisualizer

def main():
    print("="*70)
    print("CONSCIOUSNESS TRACKING - Full Analysis")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create environment
    env = GridWorld(size=10, reward_goal=1.0, reward_step=-0.01,
                   trap_positions=[(3,3), (5,5), (7,7)],  # Add traps for wisdom testing
                   goal_position=(9,9))

    # Train IDN
    print("\n1. Training Internal Dimension Network...")
    model = InternalDimensionNetwork(input_dim=2, hidden_dim=128, output_dim=4, device=device)
    trainer = PPOTrainer(model, env, device=device, use_tensorboard=False, use_wandb=False,
                        intrinsic_reward_weight=0.2, intrinsic_reward_method='balanced',
                        log_interval=20)

    history = trainer.train(num_episodes=300, steps_per_episode=200,
                           compute_consciousness_interval=10)

    # Full consciousness analysis
    print("\n2. Computing Comprehensive Consciousness Metrics...")
    print("="*70)

    consciousness_tests = ConsciousnessTests(device=device)

    # Create sample input
    sample_state = torch.FloatTensor([[0, 0]]).to(device)

    # Compute all metrics
    results = consciousness_tests.compute_overall_consciousness_score(
        model=model,
        sample_inputs=sample_state,
        run_behavioral_tests=True  # Run full test suite
    )

    # Generate report
    print("\n3. Generating Consciousness Report...")
    output_dir = Path('outputs/consciousness_tracking')
    output_dir.mkdir(parents=True, exist_ok=True)

    report = consciousness_tests.generate_consciousness_report(
        results,
        output_path=str(output_dir / 'consciousness_report.txt')
    )

    print("\n" + report)

    # Visualizations
    print("\n4. Creating Visualizations...")
    viz = InternalDimensionVisualizer()

    # Consciousness dashboard
    viz.plot_consciousness_dashboard(
        results,
        save_path=str(output_dir / 'consciousness_dashboard.png'),
        show=False
    )

    # Internal dimensions
    viz.plot_x12_m12_trajectories(
        list(model.internal_state.x12_history),
        list(model.internal_state.m12_history),
        save_path=str(output_dir / 'internal_dimensions.png'),
        show=False
    )

    # Learning curves
    viz.plot_learning_curves(
        history,
        save_path=str(output_dir / 'learning_curves.png'),
        show=False
    )

    print(f"   All visualizations saved to {output_dir}/")

    print("\n" + "="*70)
    print("CONSCIOUSNESS ANALYSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
