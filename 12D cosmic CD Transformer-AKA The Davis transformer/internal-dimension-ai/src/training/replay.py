"""
Experience Replay Buffer with Internal Dimension Priority

Implements experience replay with:
- Priority sampling based on x₁₂ (replay surprising experiences)
- m₁₂-weighted sampling (important experiences)
- Forgetting mechanism
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from collections import deque
import random


class ReplayBuffer:
    """
    Standard experience replay buffer.
    """

    def __init__(
        self,
        capacity: int = 10000,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            device: Torch device
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        **kwargs
    ):
        """
        Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            **kwargs: Additional data (x12, m12, etc.)
        """
        experience = {
            'state': state.cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu(),
            'done': done,
            **kwargs
        }

        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample random batch from buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Dictionary with batched experiences
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        return self._batch_experiences(batch)

    def _batch_experiences(self, experiences: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert list of experiences to batched tensors."""
        states = torch.cat([exp['state'] for exp in experiences]).to(self.device)
        actions = torch.tensor([exp['action'] for exp in experiences],
                              dtype=torch.long, device=self.device)
        rewards = torch.tensor([exp['reward'] for exp in experiences],
                              dtype=torch.float32, device=self.device)
        next_states = torch.cat([exp['next_state'] for exp in experiences]).to(self.device)
        dones = torch.tensor([exp['done'] for exp in experiences],
                            dtype=torch.float32, device=self.device)

        batch = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

        # Include x12 and m12 if available
        if 'x12' in experiences[0]:
            batch['x12'] = torch.tensor([exp['x12'] for exp in experiences],
                                       dtype=torch.float32, device=self.device)
        if 'm12' in experiences[0]:
            batch['m12'] = torch.tensor([exp['m12'] for exp in experiences],
                                       dtype=torch.float32, device=self.device)

        return batch

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def clear(self):
        """Clear buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay based on x₁₂ (surprise).

    Experiences with high x₁₂ (surprising) are replayed more often.
    This helps the agent learn from unexpected events.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling weight
        beta_increment: float = 0.001,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=full priority)
            beta: Importance sampling weight
            beta_increment: Increment beta each sample
            device: Torch device
        """
        super().__init__(capacity, device)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

        # Use list instead of deque to allow indexed access
        self.buffer = []
        self.priorities = []

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        x12: Optional[float] = None,
        m12: Optional[float] = None,
        **kwargs
    ):
        """
        Add experience with priority.

        Priority is based on x₁₂ (surprise/awareness level).

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            x12: Internal awareness (used for priority)
            m12: Internal memory
            **kwargs: Additional data
        """
        experience = {
            'state': state.cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu(),
            'done': done,
            'x12': x12 if x12 is not None else 0.0,
            'm12': m12 if m12 is not None else 0.0,
            **kwargs
        }

        # Priority based on |x12| (high surprise = high priority)
        if x12 is not None:
            priority = abs(x12) + 1e-6  # Small epsilon to avoid zero
        else:
            priority = self.max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # Replace oldest
            idx = len(self.buffer) - self.capacity
            self.buffer[idx] = experience
            self.priorities[idx] = priority

        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[int]]:
        """
        Sample batch with prioritization.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            batch: Batched experiences
            weights: Importance sampling weights
            indices: Sampled indices (for updating priorities)
        """
        if len(self.buffer) == 0:
            return {}, torch.tensor([]), []

        # Compute sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)]) ** self.alpha
        probabilities = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=False,
            p=probabilities
        )

        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]

        # Compute importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = self._batch_experiences(experiences)

        return batch, weights, indices.tolist()

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
            self.max_priority = max(self.max_priority, priority)


class M12WeightedReplayBuffer(PrioritizedReplayBuffer):
    """
    Replay buffer that combines x₁₂-based priority with m₁₂-based importance.

    Priority: |x₁₂| (surprise) + weight * |m₁₂| (importance from memory)

    This allows replaying both surprising events AND important remembered events.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        m12_weight: float = 0.5,  # Weight for m₁₂ in priority
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize m₁₂-weighted replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent
            beta: Importance sampling weight
            beta_increment: Beta increment per sample
            m12_weight: Weight for m₁₂ in priority calculation
            device: Torch device
        """
        super().__init__(capacity, alpha, beta, beta_increment, device)
        self.m12_weight = m12_weight

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        x12: Optional[float] = None,
        m12: Optional[float] = None,
        **kwargs
    ):
        """
        Add experience with x₁₂ and m₁₂-based priority.

        Priority = |x₁₂| + m12_weight * |m₁₂|

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            x12: Internal awareness (surprise)
            m12: Internal memory (importance)
            **kwargs: Additional data
        """
        experience = {
            'state': state.cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu(),
            'done': done,
            'x12': x12 if x12 is not None else 0.0,
            'm12': m12 if m12 is not None else 0.0,
            **kwargs
        }

        # Priority combines surprise (x12) and importance (m12)
        priority_x12 = abs(x12) if x12 is not None else 0.0
        priority_m12 = abs(m12) if m12 is not None else 0.0
        priority = priority_x12 + self.m12_weight * priority_m12 + 1e-6

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # Replace oldest
            idx = len(self.buffer) - self.capacity
            self.buffer[idx] = experience
            self.priorities[idx] = priority

        self.max_priority = max(self.max_priority, priority)


class EpisodicReplayBuffer:
    """
    Stores full episodes for episodic replay.

    Useful for:
    - Replaying complete trajectories
    - Analyzing episode-level patterns
    - Multi-step return computation
    """

    def __init__(
        self,
        capacity: int = 1000,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize episodic replay buffer.

        Args:
            capacity: Maximum number of episodes to store
            device: Torch device
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.current_episode = []

    def add_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        **kwargs
    ):
        """
        Add transition to current episode.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            **kwargs: Additional data
        """
        transition = {
            'state': state.cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu(),
            'done': done,
            **kwargs
        }

        self.current_episode.append(transition)

        if done:
            self.end_episode()

    def end_episode(self):
        """Mark current episode as complete and store it."""
        if len(self.current_episode) > 0:
            self.buffer.append(self.current_episode)
            self.current_episode = []

    def sample_episodes(self, num_episodes: int) -> List[List[Dict]]:
        """
        Sample random episodes.

        Args:
            num_episodes: Number of episodes to sample

        Returns:
            List of episodes (each episode is list of transitions)
        """
        return random.sample(self.buffer, min(num_episodes, len(self.buffer)))

    def __len__(self) -> int:
        """Return number of complete episodes."""
        return len(self.buffer)
