#!/usr/bin/env python3
"""
Run Experiment - CLI tool to run experiments

Usage:
    python scripts/run_experiment.py --exp baseline_comparison --env gridworld --episodes 1000
    python scripts/run_experiment.py --exp curiosity --env two_room --episodes 200
    python scripts/run_experiment.py --config configs/experiments/baseline_comparison.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import yaml
from src.core.network import InternalDimensionNetwork, BaselineNetwork
from src.environments.gridworld import GridWorld, TwoRoomGridWorld
from src.training.trainer import PPOTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_env(env_type: str, **kwargs):
    """Create environment."""
    if env_type == 'gridworld':
        return GridWorld(**kwargs)
    elif env_type == 'two_room':
        return TwoRoomGridWorld(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_type}")


def create_model(model_type: str, device, **kwargs):
    """Create model."""
    if model_type == 'idn':
        return InternalDimensionNetwork(device=device, **kwargs)
    elif model_type == 'baseline':
        return BaselineNetwork(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_baseline_comparison(env_config, model_config, train_config, device):
    """Run baseline comparison experiment."""
    print("Running Baseline Comparison Experiment")
    print("="*70)

    env = create_env(**env_config)

    # Train IDN
    print("\n1. Training Internal Dimension Network...")
    idn_model = create_model('idn', device, **model_config)
    idn_trainer = PPOTrainer(idn_model, env, device=device, **train_config)
    idn_history = idn_trainer.train(**train_config.get('training', {}))

    # Train Baseline
    print("\n2. Training Baseline Network...")
    baseline_model = create_model('baseline', device, **model_config)
    baseline_trainer = PPOTrainer(baseline_model, env, device=device, **train_config)
    baseline_history = baseline_trainer.train(**train_config.get('training', {}))

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")


def run_curiosity_experiment(env_config, model_config, train_config, device):
    """Run curiosity experiment."""
    print("Running Curiosity Experiment")
    print("="*70)

    env = create_env(**env_config)
    model = create_model('idn', device, **model_config)

    # Set high curiosity weight
    train_config['intrinsic_reward_weight'] = 0.5
    train_config['intrinsic_reward_method'] = 'curiosity'

    trainer = PPOTrainer(model, env, device=device, **train_config)
    history = trainer.train(**train_config.get('training', {}))

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")


def main():
    parser = argparse.ArgumentParser(description="Run Internal Dimension AI experiments")

    parser.add_argument('--exp', type=str, help='Experiment name')
    parser.add_argument('--env', type=str, default='gridworld', help='Environment type')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--config', type=str, help='Path to config YAML file')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load from config file if provided
    if args.config:
        config = load_config(args.config)
        env_config = config.get('environment', {})
        model_config = config.get('model', {})
        train_config = config.get('training', {})
        exp_name = config.get('experiment', 'custom')
    else:
        # Use command line arguments
        env_config = {'env_type': args.env, 'size': 8}
        model_config = {'input_dim': 2, 'hidden_dim': 64, 'output_dim': 4}
        train_config = {'num_episodes': args.episodes, 'use_tensorboard': False, 'use_wandb': False}
        exp_name = args.exp or 'default'

    # Run experiment
    if exp_name == 'baseline_comparison':
        run_baseline_comparison(env_config, model_config, train_config, device)
    elif exp_name == 'curiosity':
        run_curiosity_experiment(env_config, model_config, train_config, device)
    else:
        # Generic run
        print(f"Running experiment: {exp_name}")
        env = create_env(**env_config)
        model = create_model('idn', device, **model_config)
        trainer = PPOTrainer(model, env, device=device, **train_config)
        trainer.train(num_episodes=args.episodes)


if __name__ == '__main__':
    main()
