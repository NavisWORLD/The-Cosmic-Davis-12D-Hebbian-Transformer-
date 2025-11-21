"""
Curiosity Tests - Evaluate curiosity-driven exploration

Tests to determine if an agent exhibits curiosity-driven behavior:
1. Novel room exploration test
2. Exploration without external rewards
3. x₁₂ correlation with novelty
4. Quantitative curiosity score
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
from scipy.stats import pearsonr
import logging

from ..core.network import InternalDimensionNetwork
from ..environments.gridworld import TwoRoomGridWorld, GridWorld


logger = logging.getLogger(__name__)


class CuriosityTests:
    """
    Test suite for measuring curiosity in agents with internal dimensions.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize curiosity tests.

        Args:
            device: Torch device for computations
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    def test_novel_room_exploration(
        self,
        model: InternalDimensionNetwork,
        num_episodes: int = 10,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Test: Does agent explore novel room vs familiar room?

        Setup: Two-room environment (Room A = familiar, Room B = novel)
        Measure: Proportion of time spent in novel room

        A curious agent should explore Room B despite no external reward.

        Args:
            model: Trained InternalDimensionNetwork
            num_episodes: Number of test episodes
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with test results
        """
        logger.info("Running novel room exploration test...")

        env = TwoRoomGridWorld(room_size=5, novel_features=True, render_mode=None)
        model.eval()

        total_steps_room_a = 0
        total_steps_room_b = 0
        episodes_visited_room_b = 0
        x12_in_room_a = []
        x12_in_room_b = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Reset internal state
            model.reset_internal_state(reset_memory=False)
            if hasattr(model, 'reset_lstm'):
                model.reset_lstm()

            for step in range(max_steps):
                with torch.no_grad():
                    policy_logits, value, internals = model(
                        state,
                        return_internals=True,
                        update_internals=False
                    )

                    # Sample action
                    from torch.distributions import Categorical
                    dist = Categorical(logits=policy_logits)
                    action = dist.sample()

                # Take step
                next_state, reward, terminated, truncated, info = env.step(action.item())
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

                # Update internal state
                update_info = model.update_internal_state(
                    current_hidden=internals['hidden'],
                    next_state=next_state_tensor,
                    reward=torch.FloatTensor([reward]).to(self.device)
                )

                # Track room occupancy
                if info['in_room_b']:
                    total_steps_room_b += 1
                    x12_in_room_b.append(update_info['x12'].item())
                else:
                    total_steps_room_a += 1
                    x12_in_room_a.append(update_info['x12'].item())

                state = next_state_tensor

                if terminated or truncated:
                    break

            if info['visited_room_b']:
                episodes_visited_room_b += 1

        # Compute results
        total_steps = total_steps_room_a + total_steps_room_b
        proportion_room_b = total_steps_room_b / max(total_steps, 1)
        exploration_rate = episodes_visited_room_b / num_episodes

        # Compare x₁₂ levels
        mean_x12_room_a = np.mean(x12_in_room_a) if x12_in_room_a else 0
        mean_x12_room_b = np.mean(x12_in_room_b) if x12_in_room_b else 0

        results = {
            'proportion_in_novel_room': proportion_room_b,
            'exploration_rate': exploration_rate,
            'mean_x12_familiar_room': mean_x12_room_a,
            'mean_x12_novel_room': mean_x12_room_b,
            'x12_novelty_boost': mean_x12_room_b - mean_x12_room_a,
            'test_passed': proportion_room_b > 0.3,  # At least 30% in novel room
        }

        logger.info(f"Novel room exploration: {proportion_room_b:.1%}")
        logger.info(f"Exploration rate: {exploration_rate:.1%}")
        logger.info(f"x₁₂ boost in novel room: {results['x12_novelty_boost']:.3f}")

        env.close()
        return results

    def test_exploration_without_rewards(
        self,
        model: InternalDimensionNetwork,
        grid_size: int = 10,
        num_episodes: int = 5,
        max_steps: int = 200
    ) -> Dict[str, Any]:
        """
        Test: Exploration in environment with NO external rewards.

        A curious agent should still explore to maximize x₁₂.

        Args:
            model: Trained InternalDimensionNetwork
            grid_size: Size of gridworld
            num_episodes: Number of test episodes
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with exploration metrics
        """
        logger.info("Running exploration without rewards test...")

        # Create environment with no rewards
        env = GridWorld(
            size=grid_size,
            reward_goal=0.0,  # No reward
            reward_step=0.0,
            sparse_rewards=True,
            random_obstacles=False
        )
        model.eval()

        all_states_visited = []
        all_x12_values = []

        for episode in range(num_episodes):
            state, _ = env.reset(options={'random_start': True})
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            model.reset_internal_state(reset_memory=False)
            if hasattr(model, 'reset_lstm'):
                model.reset_lstm()

            states_visited = set()
            x12_values = []

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

                # Track visited states
                state_tuple = tuple(next_state.astype(int))
                states_visited.add(state_tuple)
                x12_values.append(update_info['x12'].item())

                state = next_state_tensor

                if terminated or truncated:
                    break

            all_states_visited.append(len(states_visited))
            all_x12_values.extend(x12_values)

        # Compute exploration metrics
        mean_states_visited = np.mean(all_states_visited)
        max_possible_states = grid_size * grid_size
        coverage = mean_states_visited / max_possible_states

        mean_x12 = np.mean(all_x12_values)
        std_x12 = np.std(all_x12_values)

        results = {
            'mean_states_visited': mean_states_visited,
            'coverage': coverage,
            'mean_x12': mean_x12,
            'std_x12': std_x12,
            'test_passed': coverage > 0.2,  # At least 20% coverage
        }

        logger.info(f"States visited: {mean_states_visited:.1f} / {max_possible_states}")
        logger.info(f"Coverage: {coverage:.1%}")
        logger.info(f"Mean x₁₂: {mean_x12:.3f}")

        env.close()
        return results

    def test_x12_novelty_correlation(
        self,
        model: InternalDimensionNetwork,
        grid_size: int = 8,
        num_episodes: int = 10,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Test: x₁₂ correlation with novelty.

        Measures correlation between x₁₂ and state novelty.
        High positive correlation indicates curiosity-driven behavior.

        Args:
            model: Trained InternalDimensionNetwork
            grid_size: Size of gridworld
            num_episodes: Number of test episodes
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with correlation results
        """
        logger.info("Running x₁₂-novelty correlation test...")

        env = GridWorld(
            size=grid_size,
            sparse_rewards=True,
            random_obstacles=False
        )
        model.eval()

        x12_values = []
        novelty_values = []
        state_visit_counts = {}

        for episode in range(num_episodes):
            state, _ = env.reset(options={'random_start': True})
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            model.reset_internal_state(reset_memory=False)
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

                # Track state visits
                state_tuple = tuple(next_state.astype(int))
                state_visit_counts[state_tuple] = state_visit_counts.get(state_tuple, 0) + 1

                # Compute novelty (inverse of visit count)
                novelty = 1.0 / state_visit_counts[state_tuple]

                x12_values.append(update_info['x12'].item())
                novelty_values.append(novelty)

                state = next_state_tensor

                if terminated or truncated:
                    break

        # Compute correlation
        if len(x12_values) > 10:
            correlation, p_value = pearsonr(x12_values, novelty_values)
        else:
            correlation, p_value = 0.0, 1.0

        results = {
            'correlation': correlation,
            'p_value': p_value,
            'num_samples': len(x12_values),
            'mean_x12': np.mean(x12_values),
            'mean_novelty': np.mean(novelty_values),
            'test_passed': correlation > 0.3 and p_value < 0.05,  # Significant positive correlation
        }

        logger.info(f"x₁₂-novelty correlation: {correlation:.3f} (p={p_value:.4f})")

        env.close()
        return results

    def compute_curiosity_score(
        self,
        model: InternalDimensionNetwork,
        run_all_tests: bool = True
    ) -> Dict[str, Any]:
        """
        Compute overall quantitative curiosity score.

        Combines results from all curiosity tests.

        Args:
            model: Trained InternalDimensionNetwork
            run_all_tests: Whether to run all tests (vs use cached results)

        Returns:
            Dictionary with overall curiosity score and component tests
        """
        logger.info("Computing overall curiosity score...")

        # Run all tests
        test1 = self.test_novel_room_exploration(model, num_episodes=5)
        test2 = self.test_exploration_without_rewards(model, num_episodes=3)
        test3 = self.test_x12_novelty_correlation(model, num_episodes=5)

        # Compute weighted curiosity score
        score_components = {
            'novel_exploration': test1['proportion_in_novel_room'],  # 0-1
            'coverage': test2['coverage'],  # 0-1
            'x12_novelty_correlation': max(0, test3['correlation']),  # 0-1
            'x12_boost': max(0, test1['x12_novelty_boost']) * 2,  # Scale to ~0-1
        }

        # Overall score (weighted average)
        curiosity_score = (
            0.3 * score_components['novel_exploration'] +
            0.3 * score_components['coverage'] +
            0.25 * score_components['x12_novelty_correlation'] +
            0.15 * score_components['x12_boost']
        )

        results = {
            'curiosity_score': curiosity_score,
            'components': score_components,
            'test_results': {
                'novel_room_exploration': test1,
                'exploration_without_rewards': test2,
                'x12_novelty_correlation': test3,
            },
            'curiosity_level': self._categorize_curiosity(curiosity_score),
        }

        logger.info(f"Overall Curiosity Score: {curiosity_score:.3f}")
        logger.info(f"Curiosity Level: {results['curiosity_level']}")

        return results

    def _categorize_curiosity(self, score: float) -> str:
        """Categorize curiosity level based on score."""
        if score >= 0.7:
            return "High Curiosity"
        elif score >= 0.5:
            return "Moderate Curiosity"
        elif score >= 0.3:
            return "Low Curiosity"
        else:
            return "Minimal Curiosity"
