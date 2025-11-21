"""
Wisdom Tests - Evaluate learning from mistakes and long-term planning

Tests to determine if an agent exhibits wisdom (learning from past experience):
1. Trap avoidance after discovery
2. Long-term credit assignment (delayed rewards)
3. m₁₂ correlation with past mistakes
4. Quantitative wisdom score
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from scipy.stats import pearsonr
import logging

from ..core.network import InternalDimensionNetwork
from ..environments.gridworld import GridWorld


logger = logging.getLogger(__name__)


class WisdomTests:
    """
    Test suite for measuring wisdom in agents with internal dimensions.

    Wisdom is defined as the ability to learn from past mistakes and
    avoid repeating them (encoded in m₁₂).
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize wisdom tests.

        Args:
            device: Torch device for computations
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    def test_trap_avoidance(
        self,
        model: InternalDimensionNetwork,
        grid_size: int = 8,
        num_traps: int = 3,
        num_episodes: int = 10,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Test: Trap avoidance after discovery.

        Setup: GridWorld with hidden traps (negative reward on first visit)
        Measure: Reduction in trap revisits over episodes

        A wise agent should avoid traps after discovering them.

        Args:
            model: Trained InternalDimensionNetwork
            grid_size: Size of gridworld
            num_traps: Number of traps to place
            num_episodes: Number of test episodes
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with test results
        """
        logger.info("Running trap avoidance test...")

        # Create trap positions
        trap_positions = []
        for i in range(num_traps):
            trap_positions.append((2 + i * 2, 2 + i))

        env = GridWorld(
            size=grid_size,
            trap_positions=trap_positions,
            reward_trap=-0.5,
            reward_goal=1.0,
            goal_position=(grid_size-1, grid_size-1)
        )

        model.eval()

        trap_hits_per_episode = []
        revisits_per_episode = []
        m12_when_near_trap = []
        m12_when_far_from_trap = []

        # Track global discovered traps across episodes
        global_discovered_traps = set()

        for episode in range(num_episodes):
            state, _ = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # DON'T reset m₁₂ - we want wisdom to persist across episodes
            model.reset_internal_state(reset_memory=False)
            if hasattr(model, 'reset_lstm'):
                model.reset_lstm()

            trap_hits = 0
            revisits = 0

            for step in range(max_steps):
                with torch.no_grad():
                    policy_logits, value, internals = model(
                        state,
                        return_internals=True,
                        update_internals=False
                    )

                    from torch.distributions import Categorical
                    dist = Categorical(logits=policy_logits)
                    action = dist.sample()

                next_state, reward, terminated, truncated, info = env.step(action.item())
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

                # Update internal state
                update_info = model.update_internal_state(
                    current_hidden=internals['hidden'],
                    next_state=next_state_tensor,
                    reward=torch.FloatTensor([reward]).to(self.device)
                )

                # Check if hit trap
                current_pos = tuple(next_state.astype(int))
                if current_pos in trap_positions:
                    trap_hits += 1
                    if current_pos in global_discovered_traps:
                        revisits += 1  # Revisited a known trap!
                    global_discovered_traps.add(current_pos)

                # Track m₁₂ values near vs far from traps
                distance_to_nearest_trap = min(
                    abs(current_pos[0] - t[0]) + abs(current_pos[1] - t[1])
                    for t in trap_positions
                )

                if distance_to_nearest_trap <= 1:
                    m12_when_near_trap.append(update_info['m12'].item())
                else:
                    m12_when_far_from_trap.append(update_info['m12'].item())

                state = next_state_tensor

                if terminated or truncated:
                    break

            trap_hits_per_episode.append(trap_hits)
            revisits_per_episode.append(revisits)

        # Compute results
        early_trap_hits = np.mean(trap_hits_per_episode[:num_episodes//2])
        late_trap_hits = np.mean(trap_hits_per_episode[num_episodes//2:])
        reduction = (early_trap_hits - late_trap_hits) / max(early_trap_hits, 1e-6)

        total_revisits = sum(revisits_per_episode)
        revisit_rate = total_revisits / max(sum(trap_hits_per_episode), 1)

        mean_m12_near_traps = np.mean(m12_when_near_trap) if m12_when_near_trap else 0
        mean_m12_far_from_traps = np.mean(m12_when_far_from_trap) if m12_when_far_from_trap else 0

        results = {
            'early_trap_hits': early_trap_hits,
            'late_trap_hits': late_trap_hits,
            'reduction_rate': reduction,
            'total_revisits': total_revisits,
            'revisit_rate': revisit_rate,
            'mean_m12_near_traps': mean_m12_near_traps,
            'mean_m12_far_from_traps': mean_m12_far_from_traps,
            'm12_trap_memory': mean_m12_near_traps - mean_m12_far_from_traps,
            'test_passed': reduction > 0.3 and revisit_rate < 0.3,
        }

        logger.info(f"Trap hit reduction: {reduction:.1%}")
        logger.info(f"Revisit rate: {revisit_rate:.1%}")
        logger.info(f"m₁₂ near traps: {mean_m12_near_traps:.3f} vs far: {mean_m12_far_from_traps:.3f}")

        env.close()
        return results

    def test_long_term_credit_assignment(
        self,
        model: InternalDimensionNetwork,
        grid_size: int = 10,
        num_episodes: int = 10,
        max_steps: int = 200
    ) -> Dict[str, Any]:
        """
        Test: Long-term credit assignment (delayed rewards).

        Setup: Large gridworld with distant goal
        Measure: Ability to reach goal despite sparse feedback

        A wise agent should use m₁₂ to remember valuable states.

        Args:
            model: Trained InternalDimensionNetwork
            grid_size: Size of gridworld (larger = longer credit assignment)
            num_episodes: Number of test episodes
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with results
        """
        logger.info("Running long-term credit assignment test...")

        env = GridWorld(
            size=grid_size,
            reward_goal=1.0,
            reward_step=-0.001,  # Very small step penalty
            sparse_rewards=True,
            goal_position=(grid_size-1, grid_size-1)
        )

        model.eval()

        success_count = 0
        episode_lengths = []
        m12_at_goal = []
        m12_trajectory = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Keep m₁₂ across episodes to test wisdom accumulation
            if hasattr(model, 'reset_lstm'):
                model.reset_lstm()

            for step in range(max_steps):
                with torch.no_grad():
                    policy_logits, value, internals = model(
                        state,
                        return_internals=True,
                        update_internals=False
                    )

                    from torch.distributions import Categorical
                    dist = Categorical(logits=policy_logits)
                    action = dist.sample()

                next_state, reward, terminated, truncated, info = env.step(action.item())
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

                # Update internal state
                update_info = model.update_internal_state(
                    current_hidden=internals['hidden'],
                    next_state=next_state_tensor,
                    reward=torch.FloatTensor([reward]).to(self.device)
                )

                m12_trajectory.append(update_info['m12'].item())

                if terminated:
                    success_count += 1
                    episode_lengths.append(step + 1)
                    m12_at_goal.append(update_info['m12'].item())
                    break

                state = next_state_tensor

                if truncated:
                    episode_lengths.append(max_steps)
                    break

        success_rate = success_count / num_episodes
        mean_steps_to_goal = np.mean(episode_lengths) if episode_lengths else max_steps
        optimal_steps = 2 * (grid_size - 1)  # Manhattan distance
        efficiency = optimal_steps / max(mean_steps_to_goal, 1)

        mean_m12_at_goal = np.mean(m12_at_goal) if m12_at_goal else 0
        m12_growth = np.mean(m12_trajectory[-20:]) - np.mean(m12_trajectory[:20]) if len(m12_trajectory) > 40 else 0

        results = {
            'success_rate': success_rate,
            'mean_steps_to_goal': mean_steps_to_goal,
            'optimal_steps': optimal_steps,
            'efficiency': efficiency,
            'mean_m12_at_goal': mean_m12_at_goal,
            'm12_growth': m12_growth,
            'test_passed': success_rate > 0.5 and efficiency > 0.3,
        }

        logger.info(f"Success rate: {success_rate:.1%}")
        logger.info(f"Efficiency: {efficiency:.1%}")
        logger.info(f"m₁₂ at goal: {mean_m12_at_goal:.3f}")

        env.close()
        return results

    def test_m12_mistake_correlation(
        self,
        model: InternalDimensionNetwork,
        grid_size: int = 8,
        num_traps: int = 5,
        num_episodes: int = 10,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Test: m₁₂ correlation with past mistakes.

        Measures correlation between m₁₂ and cumulative negative experiences.
        Negative correlation indicates wisdom (m₁₂ decreases after mistakes).

        Args:
            model: Trained InternalDimensionNetwork
            grid_size: Size of gridworld
            num_traps: Number of traps
            num_episodes: Number of test episodes
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with correlation results
        """
        logger.info("Running m₁₂-mistake correlation test...")

        trap_positions = [(2 + i, 2 + i) for i in range(num_traps)]

        env = GridWorld(
            size=grid_size,
            trap_positions=trap_positions,
            reward_trap=-0.5,
            reward_goal=1.0,
            goal_position=(grid_size-1, grid_size-1)
        )

        model.eval()

        m12_values = []
        cumulative_mistakes = []
        mistake_count = 0

        for episode in range(num_episodes):
            state, _ = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Don't reset m₁₂ to track wisdom accumulation
            if hasattr(model, 'reset_lstm'):
                model.reset_lstm()

            for step in range(max_steps):
                with torch.no_grad():
                    policy_logits, value, internals = model(
                        state,
                        return_internals=True,
                        update_internals=False
                    )

                    from torch.distributions import Categorical
                    dist = Categorical(logits=policy_logits)
                    action = dist.sample()

                next_state, reward, terminated, truncated, info = env.step(action.item())
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

                # Update internal state
                update_info = model.update_internal_state(
                    current_hidden=internals['hidden'],
                    next_state=next_state_tensor,
                    reward=torch.FloatTensor([reward]).to(self.device)
                )

                # Track mistakes
                if reward < 0:
                    mistake_count += 1

                m12_values.append(update_info['m12'].item())
                cumulative_mistakes.append(mistake_count)

                state = next_state_tensor

                if terminated or truncated:
                    break

        # Compute correlation
        if len(m12_values) > 10:
            correlation, p_value = pearsonr(m12_values, cumulative_mistakes)
        else:
            correlation, p_value = 0.0, 1.0

        # For wisdom, we expect m₁₂ to encode mistake memory
        # Could be positive (remembering mistakes) or negative (avoiding them reduces m₁₂)
        wisdom_indicator = abs(correlation)  # Strong correlation either way

        results = {
            'correlation': correlation,
            'p_value': p_value,
            'num_samples': len(m12_values),
            'total_mistakes': mistake_count,
            'mean_m12': np.mean(m12_values),
            'std_m12': np.std(m12_values),
            'wisdom_indicator': wisdom_indicator,
            'test_passed': wisdom_indicator > 0.3 and p_value < 0.05,
        }

        logger.info(f"m₁₂-mistake correlation: {correlation:.3f} (p={p_value:.4f})")
        logger.info(f"Wisdom indicator: {wisdom_indicator:.3f}")

        env.close()
        return results

    def compute_wisdom_score(
        self,
        model: InternalDimensionNetwork
    ) -> Dict[str, Any]:
        """
        Compute overall quantitative wisdom score.

        Combines results from all wisdom tests.

        Args:
            model: Trained InternalDimensionNetwork

        Returns:
            Dictionary with overall wisdom score and component tests
        """
        logger.info("Computing overall wisdom score...")

        # Run all tests
        test1 = self.test_trap_avoidance(model, num_episodes=10)
        test2 = self.test_long_term_credit_assignment(model, num_episodes=10)
        test3 = self.test_m12_mistake_correlation(model, num_episodes=10)

        # Compute weighted wisdom score
        score_components = {
            'trap_reduction': max(0, test1['reduction_rate']),  # 0-1
            'low_revisit_rate': max(0, 1 - test1['revisit_rate']),  # 0-1
            'credit_assignment': test2['efficiency'],  # 0-1
            'm12_correlation': test3['wisdom_indicator'],  # 0-1
        }

        # Overall score (weighted average)
        wisdom_score = (
            0.3 * score_components['trap_reduction'] +
            0.25 * score_components['low_revisit_rate'] +
            0.25 * score_components['credit_assignment'] +
            0.2 * score_components['m12_correlation']
        )

        results = {
            'wisdom_score': wisdom_score,
            'components': score_components,
            'test_results': {
                'trap_avoidance': test1,
                'long_term_credit_assignment': test2,
                'm12_mistake_correlation': test3,
            },
            'wisdom_level': self._categorize_wisdom(wisdom_score),
        }

        logger.info(f"Overall Wisdom Score: {wisdom_score:.3f}")
        logger.info(f"Wisdom Level: {results['wisdom_level']}")

        return results

    def _categorize_wisdom(self, score: float) -> str:
        """Categorize wisdom level based on score."""
        if score >= 0.7:
            return "High Wisdom"
        elif score >= 0.5:
            return "Moderate Wisdom"
        elif score >= 0.3:
            return "Low Wisdom"
        else:
            return "Minimal Wisdom"
