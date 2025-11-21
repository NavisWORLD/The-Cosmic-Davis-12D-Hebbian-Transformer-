"""
Internal Dimensions Module - Core Mathematics for x₁₂ and m₁₂

This module implements the mathematical framework for internal dimensions
based on the 12D Cosmic Synapse Theory:

- x₁₂ (Internal Awareness): Measures surprise, novelty, and attention
- m₁₂ (Accumulated Memory): Integrates experience over time

References:
    12D Cosmic Synapse Theory - Internal Dimension Consciousness Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from collections import deque


class InternalDimensionState:
    """
    Maintains the internal dimensional state (x₁₂, m₁₂) of a neural network.

    This class encapsulates all operations related to computing and updating
    the internal dimensions that give the network an "inner life".

    Attributes:
        x12 (torch.Tensor): Current awareness/surprise level ∈ [-1, 1]
        m12 (torch.Tensor): Accumulated memory/wisdom ∈ [-1, 1]
        x12_history (deque): Sliding window of past x₁₂ values
        m12_history (deque): Sliding window of past m₁₂ values
        alpha (float): Weight for prediction error in x₁₂
        beta (float): Weight for novelty in x₁₂
        gamma (float): Weight for attention in x₁₂
        eta (float): Memory integration rate for m₁₂
        delta (float): Decay rate for x₁₂
        zeta (float): Homeostatic regulation for m₁₂
    """

    def __init__(
        self,
        device: torch.device = torch.device('cpu'),
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        eta: float = 0.01,
        delta: float = 0.1,
        zeta: float = 0.001,
        history_size: int = 1000,
        m12_baseline: float = 0.0
    ):
        """
        Initialize internal dimensional state.

        Args:
            device: Torch device for computations
            alpha: Prediction error weight in x₁₂
            beta: Novelty weight in x₁₂
            gamma: Attention weight in x₁₂
            eta: Memory integration rate
            delta: x₁₂ decay rate
            zeta: m₁₂ homeostatic regulation strength
            history_size: Number of past states to maintain
            m12_baseline: Target baseline for m₁₂ homeostasis
        """
        self.device = device

        # Current internal state
        self.x12 = torch.zeros(1, device=device)
        self.m12 = torch.zeros(1, device=device)

        # History for analysis and consciousness metrics
        self.x12_history = deque(maxlen=history_size)
        self.m12_history = deque(maxlen=history_size)

        # Hyperparameters for x₁₂ dynamics
        self.alpha = alpha  # Surprise weight
        self.beta = beta    # Novelty weight
        self.gamma = gamma  # Attention weight
        self.delta = delta  # Decay rate

        # Hyperparameters for m₁₂ dynamics
        self.eta = eta      # Integration rate
        self.zeta = zeta    # Homeostatic regulation
        self.m12_baseline = m12_baseline

        # State tracking for novelty computation
        self.state_visitation_counts = {}
        self.total_states_seen = 0

    def compute_x12(
        self,
        prediction_error: torch.Tensor,
        novelty: Optional[torch.Tensor] = None,
        attention: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute x₁₂ (Internal Awareness) based on surprise, novelty, and attention.

        Mathematical formulation:
            x₁₂(t) = tanh(α·prediction_error + β·novelty + γ·attention - δ·x₁₂(t-1))

        Args:
            prediction_error: ||ŷ - y||² (squared prediction error)
            novelty: Information content I(s) of current state
            attention: Attention signal A(t)

        Returns:
            Updated x₁₂ value ∈ [-1, 1]
        """
        # Default values if not provided
        if novelty is None:
            novelty = torch.zeros_like(prediction_error)
        if attention is None:
            attention = torch.zeros_like(prediction_error)

        # Compute x₁₂ delta
        # dx₁₂/dt = α·surprise + β·novelty + γ·attention - δ·x₁₂
        x12_delta = (
            self.alpha * prediction_error +
            self.beta * novelty +
            self.gamma * attention -
            self.delta * self.x12
        )

        # Update x₁₂ with tanh normalization to keep in [-1, 1]
        self.x12 = torch.tanh(self.x12 + x12_delta)

        # Store in history
        self.x12_history.append(self.x12.item())

        return self.x12

    def update_m12(
        self,
        x12: Optional[torch.Tensor] = None,
        importance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Update m₁₂ (Accumulated Memory) by integrating x₁₂ over time.

        Mathematical formulation:
            m₁₂(t) = m₁₂(t-1) + η·x₁₂(t)·importance(t) - ζ·(m₁₂ - m₁₂_baseline)

        Args:
            x12: Current awareness level (uses self.x12 if not provided)
            importance: Salience/importance weight (e.g., |reward|)

        Returns:
            Updated m₁₂ value ∈ [-1, 1]
        """
        if x12 is None:
            x12 = self.x12
        if importance is None:
            importance = torch.ones_like(x12)

        # Integrate x₁₂ into m₁₂
        # dm₁₂/dt = η·x₁₂·importance - ζ·(m₁₂ - baseline)
        m12_delta = (
            self.eta * x12 * importance -
            self.zeta * (self.m12 - self.m12_baseline)
        )

        # Update m₁₂ with tanh normalization
        self.m12 = torch.tanh(self.m12 + m12_delta)

        # Store in history
        self.m12_history.append(self.m12.item())

        return self.m12

    def compute_prediction_error(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prediction error (surprise) as squared difference.

        Args:
            predicted: Network's prediction ŷ(t)
            actual: Actual observed value y(t)

        Returns:
            Squared prediction error ||ŷ - y||²
        """
        return torch.mean((predicted - actual) ** 2)

    def compute_novelty(
        self,
        state: torch.Tensor,
        use_count_based: bool = True
    ) -> torch.Tensor:
        """
        Compute novelty (information content) of a state.

        Two methods:
        1. Count-based: I(s) = -log(count(s) / total)
        2. Entropy-based: I(s) = H(p(s))

        Args:
            state: Current state tensor
            use_count_based: Whether to use count-based novelty

        Returns:
            Novelty score (higher = more novel)
        """
        if use_count_based:
            # Hash state for counting
            state_hash = self._hash_state(state)

            # Update visitation count
            self.state_visitation_counts[state_hash] = \
                self.state_visitation_counts.get(state_hash, 0) + 1
            self.total_states_seen += 1

            # Compute novelty: -log(p(s))
            visit_count = self.state_visitation_counts[state_hash]
            probability = visit_count / max(self.total_states_seen, 1)
            novelty = -np.log(probability + 1e-8)

            return torch.tensor(novelty, device=self.device)
        else:
            # Entropy-based novelty
            # For continuous states, use variance as proxy for entropy
            return torch.var(state)

    def compute_attention(
        self,
        hidden_state: torch.Tensor,
        method: str = 'variance'
    ) -> torch.Tensor:
        """
        Compute attention signal from network's hidden state.

        Args:
            hidden_state: Network's internal representation
            method: 'variance', 'norm', or 'entropy'

        Returns:
            Attention score (higher = more focused)
        """
        if method == 'variance':
            # High variance = high attention/activation
            return torch.var(hidden_state)
        elif method == 'norm':
            # L2 norm of activations
            return torch.norm(hidden_state)
        elif method == 'entropy':
            # Entropy of softmax over hidden states
            probs = F.softmax(hidden_state.flatten(), dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            return entropy
        else:
            raise ValueError(f"Unknown attention method: {method}")

    def get_internal_state(self) -> Dict[str, torch.Tensor]:
        """
        Get current internal dimensional state.

        Returns:
            Dictionary with x₁₂, m₁₂, and their histories
        """
        return {
            'x12': self.x12,
            'm12': self.m12,
            'x12_history': torch.tensor(list(self.x12_history), device=self.device),
            'm12_history': torch.tensor(list(self.m12_history), device=self.device)
        }

    def reset(self, reset_m12: bool = False):
        """
        Reset internal state.

        Args:
            reset_m12: Whether to also reset m₁₂ (usually keep accumulated memory)
        """
        self.x12 = torch.zeros(1, device=self.device)
        if reset_m12:
            self.m12 = torch.zeros(1, device=self.device)
            self.m12_history.clear()

        self.x12_history.clear()

    def _hash_state(self, state: torch.Tensor) -> int:
        """
        Create a hash of a state tensor for counting.

        Args:
            state: State tensor to hash

        Returns:
            Integer hash value
        """
        # Discretize continuous states for hashing
        discretized = (state.detach().cpu().numpy() * 100).astype(int)
        return hash(discretized.tobytes())

    def get_statistics(self) -> Dict[str, float]:
        """
        Compute statistics of internal dimensions over history.

        Returns:
            Dictionary with mean, std, min, max for x₁₂ and m₁₂
        """
        x12_array = np.array(self.x12_history)
        m12_array = np.array(self.m12_history)

        stats = {}

        if len(x12_array) > 0:
            stats.update({
                'x12_mean': float(np.mean(x12_array)),
                'x12_std': float(np.std(x12_array)),
                'x12_min': float(np.min(x12_array)),
                'x12_max': float(np.max(x12_array)),
                'x12_range': float(np.ptp(x12_array)),
            })

        if len(m12_array) > 0:
            stats.update({
                'm12_mean': float(np.mean(m12_array)),
                'm12_std': float(np.std(m12_array)),
                'm12_min': float(np.min(m12_array)),
                'm12_max': float(np.max(m12_array)),
                'm12_range': float(np.ptp(m12_array)),
            })

        return stats


def compute_importance(
    reward: torch.Tensor,
    method: str = 'absolute'
) -> torch.Tensor:
    """
    Compute importance weight for m₁₂ integration.

    Args:
        reward: Reward signal from environment
        method: 'absolute', 'signed', or 'squared'

    Returns:
        Importance weight
    """
    if method == 'absolute':
        return torch.abs(reward)
    elif method == 'signed':
        return reward
    elif method == 'squared':
        return reward ** 2
    else:
        raise ValueError(f"Unknown importance method: {method}")


# Convenience functions for quick usage

def create_internal_state(
    device: Optional[torch.device] = None,
    **kwargs
) -> InternalDimensionState:
    """
    Factory function to create an InternalDimensionState.

    Args:
        device: Torch device (defaults to cuda if available)
        **kwargs: Additional arguments for InternalDimensionState

    Returns:
        Initialized InternalDimensionState
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return InternalDimensionState(device=device, **kwargs)
