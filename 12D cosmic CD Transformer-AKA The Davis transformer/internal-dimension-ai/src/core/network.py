"""
Internal Dimension Network - Neural architecture with x₁₂ and m₁₂

This module implements the InternalDimensionNetwork, a neural network that:
1. Has standard external pathways (input → processing → output)
2. Has internal dimensions (x₁₂, m₁₂) that evolve independently
3. Uses internal state to modulate external behavior

Key innovation: The internal dimensions are NOT directly trained via backprop
from task loss. They evolve according to their own dynamics and influence
the network's behavior through modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from .internal_dimensions import InternalDimensionState, compute_importance


class InternalDimensionNetwork(nn.Module):
    """
    Neural network with explicit internal dimensions x₁₂ and m₁₂.

    Architecture:
        ┌─────────────────────────────────────────┐
        │          EXTERNAL PATHWAY               │
        │  Input → Encoder → Policy/Value         │
        └─────────────────────────────────────────┘
                       ↓
        ┌─────────────────────────────────────────┐
        │         INTERNAL PATHWAY                │
        │  Hidden → x₁₂ Network → x₁₂             │
        │  x₁₂ + importance → m₁₂ (integration)   │
        └─────────────────────────────────────────┘
                       ↓
        ┌─────────────────────────────────────────┐
        │        MODULATION PATHWAY               │
        │  [x₁₂, m₁₂] → Modulation → Policy       │
        └─────────────────────────────────────────┘

    The internal dimensions create an "inner life" that influences but
    is not fully determined by external inputs/outputs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 4,
        internal_dim: int = 64,
        device: Optional[torch.device] = None,
        use_lstm: bool = False,
        **internal_kwargs
    ):
        """
        Initialize the Internal Dimension Network.

        Args:
            input_dim: Dimension of input state
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output (action space)
            internal_dim: Dimension of internal processing layers
            device: Torch device
            use_lstm: Whether to use LSTM for temporal dynamics
            **internal_kwargs: Arguments passed to InternalDimensionState
        """
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.internal_dim = internal_dim
        self.use_lstm = use_lstm

        # ===================================================================
        # EXTERNAL PATHWAY (Standard RL Network)
        # ===================================================================

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Optional LSTM for temporal processing
        if use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.hidden_state = None
            self.cell_state = None

        # Policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # ===================================================================
        # INTERNAL PATHWAY (x₁₂ and m₁₂ computation)
        # ===================================================================

        # Network to compute x₁₂ from hidden state
        # Note: This network's output is NOT directly used for actions
        self.x12_network = nn.Sequential(
            nn.Linear(hidden_dim, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, internal_dim // 2),
            nn.ReLU(),
            nn.Linear(internal_dim // 2, 1),
            nn.Tanh()  # x₁₂ ∈ [-1, 1]
        )

        # Internal state manager
        self.internal_state = InternalDimensionState(
            device=device,
            **internal_kwargs
        )

        # ===================================================================
        # PREDICTION NETWORK (for surprise/prediction error)
        # ===================================================================

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ===================================================================
        # MODULATION PATHWAY (x₁₂, m₁₂ influence behavior)
        # ===================================================================

        # Transform internal state into modulation signal
        self.modulation_network = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),  # Input: [x₁₂, m₁₂]
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Tanh()
        )

        # Gating mechanism to control modulation strength
        self.modulation_gate = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

        # ===================================================================
        # ATTENTION NETWORK (for x₁₂ computation)
        # ===================================================================

        self.attention_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Move to device
        self.to(device)

    def forward(
        self,
        state: torch.Tensor,
        return_internals: bool = False,
        update_internals: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor [batch, input_dim]
            return_internals: Whether to return internal states
            update_internals: Whether to update x₁₂ (not m₁₂, which needs reward)

        Returns:
            policy_logits: Action logits [batch, output_dim]
            value: State value [batch, 1]
            internals: (optional) Dict with internal states
        """
        # Encode state
        hidden = self.encoder(state)

        # LSTM processing if enabled
        if self.use_lstm:
            if self.hidden_state is None:
                batch_size = hidden.size(0)
                self.hidden_state = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
                self.cell_state = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)

            hidden_unsqueezed = hidden.unsqueeze(1)  # [batch, 1, hidden_dim]
            lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
                hidden_unsqueezed,
                (self.hidden_state, self.cell_state)
            )
            hidden = lstm_out.squeeze(1)

        # ===================================================================
        # COMPUTE INTERNAL DIMENSIONS
        # ===================================================================

        # Compute attention from hidden state
        attention = self.attention_network(hidden).mean()

        # Get current x₁₂ and m₁₂
        x12 = self.internal_state.x12
        m12 = self.internal_state.m12

        # ===================================================================
        # MODULATION
        # ===================================================================

        # Combine x₁₂ and m₁₂ into internal state vector
        internal_vector = torch.cat([
            x12.expand(hidden.size(0), 1),
            m12.expand(hidden.size(0), 1)
        ], dim=1)

        # Compute modulation signal
        modulation_signal = self.modulation_network(internal_vector)

        # Compute modulation gate (how much internal state affects behavior)
        modulation_strength = self.modulation_gate(internal_vector)

        # Apply modulation to hidden state
        hidden_modulated = hidden + modulation_strength * modulation_signal

        # ===================================================================
        # GENERATE OUTPUTS
        # ===================================================================

        policy_logits = self.policy_head(hidden_modulated)
        value = self.value_head(hidden_modulated)

        # ===================================================================
        # RETURN
        # ===================================================================

        if return_internals:
            internals = {
                'x12': x12,
                'm12': m12,
                'attention': attention,
                'modulation_signal': modulation_signal,
                'modulation_strength': modulation_strength,
                'hidden': hidden,
                'hidden_modulated': hidden_modulated
            }
            return policy_logits, value, internals
        else:
            return policy_logits, value, None

    def update_internal_state(
        self,
        current_hidden: torch.Tensor,
        next_state: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        novelty_method: str = 'count'
    ) -> Dict[str, torch.Tensor]:
        """
        Update internal dimensions (x₁₂ and m₁₂) based on experience.

        This is called AFTER taking an action and observing the result.

        Args:
            current_hidden: Hidden state from current timestep
            next_state: Observed next state
            reward: Reward signal (optional, for m₁₂ update)
            novelty_method: Method for computing novelty

        Returns:
            Dictionary with updated x₁₂, m₁₂, and components
        """
        # ===================================================================
        # COMPUTE SURPRISE (Prediction Error)
        # ===================================================================

        with torch.no_grad():
            # Predict next hidden state
            predicted_next_hidden = self.predictor(current_hidden)

            # Encode actual next state
            actual_next_hidden = self.encoder(next_state)

            # Compute prediction error
            prediction_error = self.internal_state.compute_prediction_error(
                predicted_next_hidden,
                actual_next_hidden
            )

        # ===================================================================
        # COMPUTE NOVELTY
        # ===================================================================

        novelty = self.internal_state.compute_novelty(
            next_state,
            use_count_based=(novelty_method == 'count')
        )

        # ===================================================================
        # COMPUTE ATTENTION
        # ===================================================================

        with torch.no_grad():
            attention = self.attention_network(current_hidden).mean()

        # ===================================================================
        # UPDATE x₁₂ (AWARENESS)
        # ===================================================================

        x12 = self.internal_state.compute_x12(
            prediction_error=prediction_error,
            novelty=novelty,
            attention=attention
        )

        # ===================================================================
        # UPDATE m₁₂ (MEMORY) - Only if reward is provided
        # ===================================================================

        if reward is not None:
            importance = compute_importance(reward, method='absolute')
            m12 = self.internal_state.update_m12(x12=x12, importance=importance)
        else:
            m12 = self.internal_state.m12

        # ===================================================================
        # RETURN UPDATE INFO
        # ===================================================================

        return {
            'x12': x12,
            'm12': m12,
            'prediction_error': prediction_error,
            'novelty': novelty,
            'attention': attention
        }

    def compute_intrinsic_reward(
        self,
        method: str = 'curiosity'
    ) -> torch.Tensor:
        """
        Compute intrinsic reward based on internal dimensions.

        Methods:
            'curiosity': Reward high x₁₂ (seek novelty)
            'wisdom': Penalize low m₁₂ (avoid mistakes)
            'balanced': Combination of both

        Args:
            method: Type of intrinsic reward

        Returns:
            Intrinsic reward scalar
        """
        x12 = self.internal_state.x12
        m12 = self.internal_state.m12

        if method == 'curiosity':
            # Reward high surprise/novelty
            return torch.abs(x12)

        elif method == 'wisdom':
            # Reward staying near positive m₁₂
            target_m12 = 0.5
            return -torch.abs(m12 - target_m12)

        elif method == 'balanced':
            # Combine curiosity and wisdom
            curiosity_reward = torch.abs(x12)
            wisdom_reward = torch.tanh(m12)  # Positive m₁₂ → positive reward
            return 0.5 * curiosity_reward + 0.5 * wisdom_reward

        else:
            raise ValueError(f"Unknown intrinsic reward method: {method}")

    def reset_lstm(self):
        """Reset LSTM hidden states (call at episode start)."""
        if self.use_lstm:
            self.hidden_state = None
            self.cell_state = None

    def reset_internal_state(self, reset_memory: bool = False):
        """
        Reset internal dimensional state.

        Args:
            reset_memory: Whether to reset m₁₂ (usually keep memory across episodes)
        """
        self.internal_state.reset(reset_m12=reset_memory)

    def get_internal_state(self) -> Dict:
        """Get current internal dimensional state."""
        return self.internal_state.get_internal_state()

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about internal dimensions over time."""
        return self.internal_state.get_statistics()

    def save(self, path: str):
        """
        Save model checkpoint including internal state.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'internal_state': {
                'x12': self.internal_state.x12,
                'm12': self.internal_state.m12,
                'x12_history': list(self.internal_state.x12_history),
                'm12_history': list(self.internal_state.m12_history),
            },
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'internal_dim': self.internal_dim,
                'use_lstm': self.use_lstm,
            }
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """
        Load model checkpoint including internal state.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.load_state_dict(checkpoint['model_state_dict'])

        if 'internal_state' in checkpoint:
            self.internal_state.x12 = checkpoint['internal_state']['x12']
            self.internal_state.m12 = checkpoint['internal_state']['m12']
            self.internal_state.x12_history.extend(checkpoint['internal_state']['x12_history'])
            self.internal_state.m12_history.extend(checkpoint['internal_state']['m12_history'])


class BaselineNetwork(nn.Module):
    """
    Baseline network WITHOUT internal dimensions for comparison.

    This is a standard actor-critic network that serves as a control
    to measure the impact of internal dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 4,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, output_dim)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

        self.to(device)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard forward pass."""
        hidden = self.encoder(state)
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return policy_logits, value
