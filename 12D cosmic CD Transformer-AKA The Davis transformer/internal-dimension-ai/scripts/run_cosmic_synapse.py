#!/usr/bin/env python3
"""
Cosmic Synapse Integration - Physics-Conditioned Transformer

Runs the full 12D physics + transformer co-evolution experiment.
This is the advanced version that couples physical dynamics with
language model training.

Usage:
    python scripts/run_cosmic_synapse.py [--steps STEPS] [--particles N]

Example:
    python scripts/run_cosmic_synapse.py --steps 10000 --particles 64
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.advanced.cosmic_synapse import run_experiment


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Cosmic Synapse physics-transformer integration'
    )

    parser.add_argument(
        '--particles',
        type=int,
        default=256,
        help='Number of particles in 12D physics simulation (default: 256)'
    )

    parser.add_argument(
        '--physics-steps',
        type=int,
        default=10000,
        help='Initial physics burn-in steps (default: 10000)'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=2000000,
        help='Total training steps (default: 2000000)'
    )

    parser.add_argument(
        '--physics-per-train',
        type=int,
        default=1,
        help='Physics steps per training step (default: 1)'
    )

    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=200000,
        help='Steps between checkpoints (default: 200000)'
    )

    parser.add_argument(
        '--generation-interval',
        type=int,
        default=200000,
        help='Steps between text generation samples (default: 200000)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Running Cosmic Synapse experiment with:")
    print(f"  Particles: {args.particles}")
    print(f"  Physics burn-in: {args.physics_steps} steps")
    print(f"  Training steps: {args.steps}")
    print(f"  Physics per train: {args.physics_per_train}")
    print(f"  Device: {device}")
    print()

    # Run experiment
    run_experiment(
        n_particles=args.particles,
        physics_steps=args.physics_steps,
        train_steps=args.steps,
        physics_steps_per_train=args.physics_per_train,
        checkpoint_interval=args.checkpoint_interval,
        generation_interval=args.generation_interval,
        device=device
    )
