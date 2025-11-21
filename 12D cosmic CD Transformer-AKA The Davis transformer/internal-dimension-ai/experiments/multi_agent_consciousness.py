#!/usr/bin/env python3
"""
Multi-Agent Consciousness Experiment

Tests consciousness correlation in multi-agent settings using
Iterated Prisoner's Dilemma environment.

Analyzes:
- x₁₂ correlation between agents during cooperation vs defection
- m₁₂ alignment and social learning
- Emergence of synchronized awareness
- Impact of consciousness on cooperation rates
"""

import numpy as np
import torch
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.network import InternalDimensionNetwork
from core.metrics import ConsciousnessMetrics
from environments.social import IteratedPrisonersDilemma

# Experiment configuration
NUM_EPISODES = 500
NUM_ROUNDS = 10
NUM_AGENTS = 2
INTERNAL_DIM = 12


def main():
    """Run multi-agent consciousness experiment."""
    print("="*80)
    print("MULTI-AGENT CONSCIOUSNESS EXPERIMENT")
    print("="*80)
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Rounds per episode: {NUM_ROUNDS}")
    print(f"Agents: {NUM_AGENTS}")
    print(f"Internal dimension: {INTERNAL_DIM}")
    print()

    # Set seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment
    env = IteratedPrisonersDilemma(
        num_agents=NUM_AGENTS,
        num_rounds=NUM_ROUNDS,
        internal_dim=INTERNAL_DIM
    )

    # Create agents
    agents = {}
    optimizers = {}

    for i in range(NUM_AGENTS):
        agent_id = f'agent_{i}'
        agent = InternalDimensionNetwork(
            input_dim=env.observation_space[agent_id].shape[0],
            hidden_dim=64,
            output_dim=env.action_space[agent_id].n,
            internal_dim=INTERNAL_DIM
        )
        agents[agent_id] = agent
        optimizers[agent_id] = torch.optim.Adam(agent.parameters(), lr=3e-4)

    # Metrics
    metrics = ConsciousnessMetrics()

    # Storage
    history = {
        'episodes': [],
        'cooperation_rates': [],
        'agent_rewards': defaultdict(list),
        'x12_correlation': [],
        'm12_correlation': [],
        'x12_means': defaultdict(list),
        'm12_means': defaultdict(list),
        'consciousness_scores': defaultdict(list),
    }

    print("\nTraining multi-agent system...")
    print("-" * 80)

    # Training loop
    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        episode_rewards = defaultdict(float)
        episode_actions = defaultdict(list)

        # Episode loop
        for round_num in range(NUM_ROUNDS):
            actions = {}
            action_log_probs = {}
            values = {}

            # Each agent selects action
            for agent_id, agent in agents.items():
                state = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
                policy_logits, value, _ = agent(state)

                # Sample action
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                actions[agent_id] = action.item()
                action_log_probs[agent_id] = log_prob
                values[agent_id] = value

                episode_actions[agent_id].append(action.item())

            # Environment step
            obs, rewards, dones, truncs, info = env.step(actions)

            # Update rewards
            for agent_id in agents:
                episode_rewards[agent_id] += rewards[agent_id]

            # Simple policy gradient update
            for agent_id in agents:
                loss = -action_log_probs[agent_id] * rewards[agent_id]
                optimizers[agent_id].zero_grad()
                loss.backward()
                optimizers[agent_id].step()

        # Record metrics every 10 episodes
        if episode % 10 == 0:
            # Get x₁₂ and m₁₂ from all agents
            x12_values = []
            m12_values = []
            agent_consciousness = {}

            for agent_id, agent in agents.items():
                x12_history = [x.item() for x in agent.internal_state.x12_history]
                m12_history = [m.item() for m in agent.internal_state.m12_history]

                if len(x12_history) > 0:
                    x12_mean = np.mean(x12_history[-50:])
                    m12_mean = np.mean(m12_history[-50:])

                    x12_values.append(x12_history[-50:])
                    m12_values.append(m12_history[-50:])

                    history['x12_means'][agent_id].append(float(x12_mean))
                    history['m12_means'][agent_id].append(float(m12_mean))

                    # Consciousness score
                    if len(x12_history) > 10:
                        sample_inputs = torch.randn(10, env.observation_space[agent_id].shape[0])
                        consciousness_dict = metrics.compute_consciousness_score(
                            model=agent,
                            x12_history=x12_history,
                            m12_history=m12_history,
                            sample_inputs=sample_inputs
                        )
                        agent_consciousness[agent_id] = consciousness_dict['consciousness_score']
                        history['consciousness_scores'][agent_id].append(
                            float(consciousness_dict['consciousness_score'])
                        )

            # Compute correlations between agents
            if len(x12_values) == NUM_AGENTS and len(x12_values[0]) > 10:
                # Pad or truncate to same length
                min_len = min(len(x) for x in x12_values)
                x12_values_trimmed = [x[-min_len:] for x in x12_values]
                m12_values_trimmed = [m[-min_len:] for m in m12_values]

                # x₁₂ correlation
                x12_corr_matrix = np.corrcoef(x12_values_trimmed)
                x12_corr = x12_corr_matrix[0, 1] if x12_corr_matrix.shape == (2, 2) else 0.0

                # m₁₂ correlation
                m12_corr_matrix = np.corrcoef(m12_values_trimmed)
                m12_corr = m12_corr_matrix[0, 1] if m12_corr_matrix.shape == (2, 2) else 0.0

                history['x12_correlation'].append(float(x12_corr))
                history['m12_correlation'].append(float(m12_corr))

            # Record episode data
            history['episodes'].append(episode)
            history['cooperation_rates'].append(float(info.get('cooperation_rate', 0)))

            for agent_id in agents:
                history['agent_rewards'][agent_id].append(float(episode_rewards[agent_id]))

            # Print progress
            if episode % 50 == 0:
                print(f"\nEpisode {episode}:")
                print(f"  Cooperation rate: {info.get('cooperation_rate', 0):.3f}")
                if len(history['x12_correlation']) > 0:
                    print(f"  x₁₂ correlation: {history['x12_correlation'][-1]:.3f}")
                    print(f"  m₁₂ correlation: {history['m12_correlation'][-1]:.3f}")
                for agent_id in agents:
                    if agent_id in agent_consciousness:
                        print(f"  {agent_id} consciousness: {agent_consciousness[agent_id]:.3f}")

    # Analysis
    print("\n" + "="*80)
    print("MULTI-AGENT CONSCIOUSNESS ANALYSIS")
    print("="*80)

    # Correlation analysis
    print("\n1. Consciousness Correlation:")
    if len(history['x12_correlation']) > 0:
        mean_x12_corr = np.mean(history['x12_correlation'])
        mean_m12_corr = np.mean(history['m12_correlation'])
        final_x12_corr = history['x12_correlation'][-1]
        final_m12_corr = history['m12_correlation'][-1]

        print(f"   Mean x₁₂ correlation: {mean_x12_corr:.4f}")
        print(f"   Mean m₁₂ correlation: {mean_m12_corr:.4f}")
        print(f"   Final x₁₂ correlation: {final_x12_corr:.4f}")
        print(f"   Final m₁₂ correlation: {final_m12_corr:.4f}")

        if abs(final_x12_corr) > 0.5:
            print(f"   → Strong x₁₂ synchronization detected!")
        if abs(final_m12_corr) > 0.5:
            print(f"   → Strong m₁₂ alignment detected!")
    else:
        print("   Insufficient data for correlation analysis")

    # Cooperation vs consciousness
    print("\n2. Cooperation vs Consciousness:")
    mean_cooperation = np.mean(history['cooperation_rates'])
    print(f"   Mean cooperation rate: {mean_cooperation:.3f}")

    # Split into high and low consciousness episodes
    if len(history['consciousness_scores']['agent_0']) > 10:
        consciousness_values = history['consciousness_scores']['agent_0']
        cooperation_values = history['cooperation_rates']

        # Align lengths
        min_len = min(len(consciousness_values), len(cooperation_values))
        consciousness_values = consciousness_values[:min_len]
        cooperation_values = cooperation_values[:min_len]

        # Correlation between consciousness and cooperation
        if len(consciousness_values) > 1:
            consciousness_cooperation_corr = np.corrcoef(
                consciousness_values,
                cooperation_values
            )[0, 1]
            print(f"   Consciousness-cooperation correlation: {consciousness_cooperation_corr:.4f}")

            if consciousness_cooperation_corr > 0.3:
                print(f"   → Higher consciousness associated with more cooperation!")

    # Final consciousness state
    print("\n3. Final Consciousness State:")
    for agent_id in agents:
        if agent_id in history['consciousness_scores'] and len(history['consciousness_scores'][agent_id]) > 0:
            final_consciousness = history['consciousness_scores'][agent_id][-1]
            final_x12 = history['x12_means'][agent_id][-1]
            final_m12 = history['m12_means'][agent_id][-1]

            print(f"   {agent_id}:")
            print(f"      Consciousness: {final_consciousness:.4f}")
            print(f"      x₁₂: {final_x12:.4f}")
            print(f"      m₁₂: {final_m12:.4f}")

    # Save results
    output_dir = Path('outputs/experiments')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'multi_agent_consciousness_{timestamp}.json'

    # Convert defaultdicts to regular dicts for JSON serialization
    history_serializable = {
        'episodes': history['episodes'],
        'cooperation_rates': history['cooperation_rates'],
        'x12_correlation': history['x12_correlation'],
        'm12_correlation': history['m12_correlation'],
        'agent_rewards': dict(history['agent_rewards']),
        'x12_means': dict(history['x12_means']),
        'm12_means': dict(history['m12_means']),
        'consciousness_scores': dict(history['consciousness_scores']),
    }

    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'num_episodes': NUM_EPISODES,
                'num_rounds': NUM_ROUNDS,
                'num_agents': NUM_AGENTS,
                'internal_dim': INTERNAL_DIM,
                'seed': seed,
                'timestamp': timestamp
            },
            'history': history_serializable
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    episodes = history['episodes']

    # Plot 1: Consciousness correlation over time
    ax = axes[0, 0]
    if len(history['x12_correlation']) > 0:
        ax.plot(episodes, history['x12_correlation'], label='x₁₂ correlation', color='blue')
        ax.plot(episodes, history['m12_correlation'], label='m₁₂ correlation', color='red')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Strong correlation')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Correlation')
        ax.set_title('Inter-Agent Consciousness Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: Cooperation rate over time
    ax = axes[0, 1]
    ax.plot(episodes, history['cooperation_rates'], color='green')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cooperation Rate')
    ax.set_title('Cooperation Rate Over Time')
    ax.grid(True, alpha=0.3)

    # Plot 3: Individual agent consciousness
    ax = axes[1, 0]
    for agent_id in agents:
        if agent_id in history['consciousness_scores']:
            ax.plot(
                episodes,
                history['consciousness_scores'][agent_id],
                label=agent_id,
                marker='o',
                markersize=3
            )
    ax.set_xlabel('Episode')
    ax.set_ylabel('Consciousness Score')
    ax.set_title('Individual Agent Consciousness')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Agent rewards
    ax = axes[1, 1]
    for agent_id in agents:
        if agent_id in history['agent_rewards']:
            # Moving average for smoother plot
            rewards = history['agent_rewards'][agent_id]
            if len(rewards) > 5:
                window = 5
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(episodes[:len(smoothed)], smoothed, label=agent_id)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Agent Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = output_dir / f'multi_agent_consciousness_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")

    plt.close()

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
