"""
Sequence Prediction Environment

Tests pattern learning and surprise detection:
- Present sequences (e.g., A-B-C-D, A-B-C-?)
- Agent must predict next element
- Surprise (x₁₂) should spike on unexpected sequences
- Measures learning and expectation formation
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any, List


class SequencePredictionEnv(gym.Env):
    """
    Sequence prediction environment for testing pattern learning.

    The agent observes sequences and must predict the next element.
    Useful for testing:
    - Pattern recognition
    - Expectation formation
    - Surprise detection (x₁₂ spikes on violations)
    """

    metadata = {'render_modes': ['ascii'], 'render_fps': 4}

    def __init__(
        self,
        sequence_length: int = 4,
        num_symbols: int = 5,
        pattern_type: str = 'sequential',  # 'sequential', 'alternating', 'random'
        reward_correct: float = 1.0,
        reward_incorrect: float = -0.5,
        max_steps: int = 100,
        render_mode: Optional[str] = None
    ):
        """
        Initialize sequence prediction environment.

        Args:
            sequence_length: Length of sequences to present
            num_symbols: Number of unique symbols
            pattern_type: Type of pattern ('sequential', 'alternating', 'random')
            reward_correct: Reward for correct prediction
            reward_incorrect: Penalty for incorrect prediction
            max_steps: Maximum steps per episode
            render_mode: Rendering mode
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.num_symbols = num_symbols
        self.pattern_type = pattern_type
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Action space: predict next symbol (0 to num_symbols-1)
        self.action_space = spaces.Discrete(num_symbols)

        # Observation space: one-hot encoding of current sequence
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(sequence_length * num_symbols,),
            dtype=np.float32
        )

        # State
        self.current_sequence = []
        self.target_symbol = None
        self.steps = 0
        self.correct_predictions = 0

    def _generate_sequence(self) -> Tuple[List[int], int]:
        """
        Generate a sequence and target symbol.

        Returns:
            sequence: List of symbols
            target: Next symbol to predict
        """
        if self.pattern_type == 'sequential':
            # Simple sequential pattern: 0,1,2,3,... → 4
            start = np.random.randint(0, self.num_symbols - self.sequence_length)
            sequence = list(range(start, start + self.sequence_length))
            target = (start + self.sequence_length) % self.num_symbols

        elif self.pattern_type == 'alternating':
            # Alternating pattern: A,B,A,B,... → A
            symbols = np.random.choice(self.num_symbols, size=2, replace=False)
            sequence = [symbols[i % 2] for i in range(self.sequence_length)]
            target = symbols[self.sequence_length % 2]

        elif self.pattern_type == 'random':
            # Random sequences
            sequence = np.random.choice(self.num_symbols, size=self.sequence_length).tolist()
            target = np.random.randint(0, self.num_symbols)

        elif self.pattern_type == 'repeat':
            # Repeat last element
            sequence = np.random.choice(self.num_symbols, size=self.sequence_length).tolist()
            target = sequence[-1]

        else:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")

        return sequence, target

    def _sequence_to_observation(self, sequence: List[int]) -> np.ndarray:
        """
        Convert sequence to one-hot encoded observation.

        Args:
            sequence: List of symbols

        Returns:
            One-hot encoded observation
        """
        obs = np.zeros(self.sequence_length * self.num_symbols, dtype=np.float32)

        for i, symbol in enumerate(sequence):
            obs[i * self.num_symbols + symbol] = 1.0

        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        self.current_sequence, self.target_symbol = self._generate_sequence()
        self.steps = 0
        self.correct_predictions = 0

        obs = self._sequence_to_observation(self.current_sequence)

        return obs, {'target': self.target_symbol}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a prediction step.

        Args:
            action: Predicted next symbol

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1

        # Check if prediction is correct
        if action == self.target_symbol:
            reward = self.reward_correct
            correct = True
            self.correct_predictions += 1
        else:
            reward = self.reward_incorrect
            correct = False

        # Generate new sequence
        self.current_sequence, self.target_symbol = self._generate_sequence()
        obs = self._sequence_to_observation(self.current_sequence)

        # Episode termination
        terminated = False
        truncated = self.steps >= self.max_steps

        info = {
            'correct': correct,
            'target': self.target_symbol,
            'prediction': action,
            'accuracy': self.correct_predictions / max(self.steps, 1),
            'steps': self.steps
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == 'ascii':
            symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            seq_str = '-'.join([symbols[s] for s in self.current_sequence])
            target_str = symbols[self.target_symbol]

            print(f"Sequence: {seq_str} → ?")
            print(f"Target: {target_str}")
            print(f"Accuracy: {self.correct_predictions}/{self.steps}")
            print()

    def close(self):
        """Close environment."""
        pass


class PatternViolationEnv(gym.Env):
    """
    Environment that occasionally violates established patterns.

    Used to test surprise detection:
    - Most sequences follow a pattern
    - Occasionally present violations
    - x₁₂ should spike on violations
    """

    metadata = {'render_modes': ['ascii'], 'render_fps': 4}

    def __init__(
        self,
        num_symbols: int = 4,
        violation_probability: float = 0.1,
        reward_correct: float = 1.0,
        reward_incorrect: float = -0.5,
        max_steps: int = 100,
        render_mode: Optional[str] = None
    ):
        """
        Initialize pattern violation environment.

        Args:
            num_symbols: Number of unique symbols
            violation_probability: Probability of violating pattern
            reward_correct: Reward for correct prediction
            reward_incorrect: Penalty for incorrect prediction
            max_steps: Maximum steps per episode
            render_mode: Rendering mode
        """
        super().__init__()

        self.num_symbols = num_symbols
        self.violation_probability = violation_probability
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Establish pattern: 0→1, 1→2, 2→3, 3→0
        self.pattern_map = {i: (i + 1) % num_symbols for i in range(num_symbols)}

        self.action_space = spaces.Discrete(num_symbols)
        self.observation_space = spaces.Box(
            low=0,
            high=num_symbols - 1,
            shape=(1,),
            dtype=np.float32
        )

        self.current_symbol = 0
        self.next_symbol = None
        self.is_violation = False
        self.steps = 0
        self.violations_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        self.current_symbol = np.random.randint(0, self.num_symbols)
        self.steps = 0
        self.violations_count = 0

        self._compute_next_symbol()

        return np.array([self.current_symbol], dtype=np.float32), {}

    def _compute_next_symbol(self):
        """Compute next symbol (following pattern or violating)."""
        if np.random.rand() < self.violation_probability:
            # Violation: choose random symbol (not the expected one)
            expected = self.pattern_map[self.current_symbol]
            possible = [s for s in range(self.num_symbols) if s != expected]
            self.next_symbol = np.random.choice(possible)
            self.is_violation = True
            self.violations_count += 1
        else:
            # Follow pattern
            self.next_symbol = self.pattern_map[self.current_symbol]
            self.is_violation = False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Predict next symbol.

        Args:
            action: Predicted next symbol

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps += 1

        # Check prediction
        if action == self.next_symbol:
            reward = self.reward_correct
            correct = True
        else:
            reward = self.reward_incorrect
            correct = False

        # Move to next symbol
        self.current_symbol = self.next_symbol
        self._compute_next_symbol()

        obs = np.array([self.current_symbol], dtype=np.float32)

        terminated = False
        truncated = self.steps >= self.max_steps

        info = {
            'correct': correct,
            'was_violation': self.is_violation,
            'violations_count': self.violations_count,
            'steps': self.steps
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render environment."""
        if self.render_mode == 'ascii':
            symbols = ['A', 'B', 'C', 'D']
            current = symbols[self.current_symbol]
            expected = symbols[self.pattern_map[self.current_symbol]]

            print(f"Current: {current}")
            print(f"Expected next: {expected}")
            print(f"Violations so far: {self.violations_count}/{self.steps}")
            print()

    def close(self):
        """Close environment."""
        pass
