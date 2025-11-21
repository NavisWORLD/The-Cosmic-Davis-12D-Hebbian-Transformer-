#!/usr/bin/env python3
"""
Evaluate Consciousness Metrics for Trained Agents

This script loads trained agents and evaluates their consciousness-related metrics
including R_ω (internal dimension richness), R_ψ (phenomenal binding), and φ
(integrated information approximation).

Usage:
    python scripts/evaluate_consciousness.py --checkpoint checkpoints/agent.pt --episodes 100
    python scripts/evaluate_consciousness.py --checkpoint checkpoints/agent.pt --output results/consciousness.json

Examples:
    # Basic evaluation
    python scripts/evaluate_consciousness.py --checkpoint checkpoints/best_agent.pt

    # With specific environment
    python scripts/evaluate_consciousness.py --checkpoint checkpoints/agent.pt --env TwoRoomGridWorld

    # Save detailed results
    python scripts/evaluate_consciousness.py --checkpoint checkpoints/agent.pt --output results/eval.json --verbose
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.ppo import PPOAgent
from core.consciousness import ConsciousnessMetrics
from environments.gridworld import GridWorld, TwoRoomGridWorld
from environments.sparse_reward import SparseRewardWrapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path]) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file, or None for defaults

    Returns:
        Configuration dictionary
    """
    if config_path and config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Default configuration
    return {
        "environment": {
            "name": "TwoRoomGridWorld",
            "size": 8,
            "num_rooms": 2,
        },
        "agent": {
            "hidden_size": 128,
            "internal_dim": 12,
        },
    }


def create_environment(env_config: Dict):
    """Create environment from configuration.

    Args:
        env_config: Environment configuration dictionary

    Returns:
        Gymnasium environment
    """
    env_name = env_config.get("name", "GridWorld")

    if env_name == "GridWorld":
        env = GridWorld(size=env_config.get("size", 5))
    elif env_name == "TwoRoomGridWorld":
        env = TwoRoomGridWorld(
            size=env_config.get("size", 8),
            num_rooms=env_config.get("num_rooms", 2),
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    # Apply wrappers if specified
    if env_config.get("sparse_reward", False):
        env = SparseRewardWrapper(env, sparsity=env_config.get("sparsity", 0.9))

    return env


def load_agent(checkpoint_path: Path, config: Dict, env) -> PPOAgent:
    """Load agent from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        env: Environment instance

    Returns:
        Loaded PPOAgent
    """
    agent_config = config["agent"]

    agent = PPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_size=agent_config.get("hidden_size", 128),
        internal_dim=agent_config.get("internal_dim", 12),
        learning_rate=agent_config.get("learning_rate", 3e-4),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    if isinstance(checkpoint, dict) and 'agent_state_dict' in checkpoint:
        agent.policy.load_state_dict(checkpoint['agent_state_dict'])
        logger.info(f"Loaded agent from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        agent.policy.load_state_dict(checkpoint)
        logger.info("Loaded agent state dict")

    agent.policy.eval()
    return agent


def evaluate_episode(
    agent: PPOAgent,
    env,
    consciousness_metrics: ConsciousnessMetrics,
    max_steps: int = 500,
) -> Tuple[float, Dict[str, List[float]]]:
    """Evaluate agent for one episode and collect consciousness metrics.

    Args:
        agent: PPOAgent to evaluate
        env: Environment instance
        consciousness_metrics: ConsciousnessMetrics instance
        max_steps: Maximum steps per episode

    Returns:
        Tuple of (total_reward, metrics_dict)
    """
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    step = 0

    # Track metrics over episode
    x12_values = []
    m12_values = []
    actions_taken = []

    while not done and step < max_steps:
        # Get action from agent
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_logits, value, x12, m12 = agent.policy(obs_tensor)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()

        # Store internal dimensions
        x12_values.append(x12.squeeze(0).numpy())
        m12_values.append(m12.squeeze(0).numpy())
        actions_taken.append(action.item())

        # Take step
        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward
        step += 1

    # Convert to numpy arrays
    x12_array = np.array(x12_values)  # (T, internal_dim)
    m12_array = np.array(m12_values)  # (T, internal_dim)
    actions_array = np.array(actions_taken)  # (T,)

    # Compute consciousness metrics
    R_omega = consciousness_metrics.compute_R_omega(x12_array)
    R_psi = consciousness_metrics.compute_R_psi(m12_array, actions_array)
    phi = consciousness_metrics.compute_phi(x12_array)

    metrics = {
        "R_omega": [R_omega],
        "R_psi": [R_psi],
        "phi": [phi],
        "episode_length": [step],
        "x12_mean_norm": [np.linalg.norm(x12_array.mean(axis=0))],
        "m12_mean_norm": [np.linalg.norm(m12_array.mean(axis=0))],
        "x12_variance": [x12_array.var()],
        "m12_variance": [m12_array.var()],
    }

    return total_reward, metrics


def evaluate_agent(
    agent: PPOAgent,
    env,
    num_episodes: int = 100,
    max_steps: int = 500,
    verbose: bool = False,
) -> Dict:
    """Evaluate agent over multiple episodes.

    Args:
        agent: PPOAgent to evaluate
        env: Environment instance
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        verbose: Print detailed progress

    Returns:
        Dictionary of evaluation results
    """
    consciousness_metrics = ConsciousnessMetrics(internal_dim=agent.internal_dim)

    all_rewards = []
    all_metrics = {
        "R_omega": [],
        "R_psi": [],
        "phi": [],
        "episode_length": [],
        "x12_mean_norm": [],
        "m12_mean_norm": [],
        "x12_variance": [],
        "m12_variance": [],
    }

    logger.info(f"Evaluating agent for {num_episodes} episodes...")

    for episode in range(num_episodes):
        reward, metrics = evaluate_episode(
            agent, env, consciousness_metrics, max_steps
        )

        all_rewards.append(reward)
        for key, values in metrics.items():
            all_metrics[key].extend(values)

        if verbose and (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode + 1}/{num_episodes}: "
                       f"Reward={reward:.2f}, "
                       f"R_ω={metrics['R_omega'][0]:.4f}, "
                       f"R_ψ={metrics['R_psi'][0]:.4f}, "
                       f"φ={metrics['phi'][0]:.4f}")

    # Compute statistics
    results = {
        "num_episodes": num_episodes,
        "reward": {
            "mean": float(np.mean(all_rewards)),
            "std": float(np.std(all_rewards)),
            "min": float(np.min(all_rewards)),
            "max": float(np.max(all_rewards)),
        },
        "consciousness": {
            metric: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for metric, values in all_metrics.items()
        },
    }

    return results


def print_results(results: Dict):
    """Print evaluation results in a formatted way.

    Args:
        results: Results dictionary from evaluate_agent
    """
    print("\n" + "=" * 60)
    print("CONSCIOUSNESS EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nEpisodes: {results['num_episodes']}")

    print("\n--- PERFORMANCE ---")
    print(f"Reward: {results['reward']['mean']:.2f} ± {results['reward']['std']:.2f}")
    print(f"  Range: [{results['reward']['min']:.2f}, {results['reward']['max']:.2f}]")

    print("\n--- CONSCIOUSNESS METRICS ---")
    for metric in ["R_omega", "R_psi", "phi"]:
        stats = results['consciousness'][metric]
        print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    print("\n--- INTERNAL DIMENSION STATISTICS ---")
    for metric in ["x12_mean_norm", "m12_mean_norm", "x12_variance", "m12_variance"]:
        stats = results['consciousness'][metric]
        print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate consciousness metrics for trained agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to agent checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Environment name (overrides config)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save results JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_config(args.config)

    # Override environment if specified
    if args.env:
        config["environment"]["name"] = args.env

    # Create environment
    env = create_environment(config["environment"])
    logger.info(f"Created environment: {config['environment']['name']}")

    # Load agent
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return 1

    agent = load_agent(args.checkpoint, config, env)
    logger.info(f"Loaded agent from: {args.checkpoint}")

    # Evaluate agent
    results = evaluate_agent(
        agent,
        env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    # Print results
    print_results(results)

    # Save results if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
