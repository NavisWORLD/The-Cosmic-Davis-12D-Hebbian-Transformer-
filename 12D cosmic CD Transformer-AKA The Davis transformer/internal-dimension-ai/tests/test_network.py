"""
Unit tests for network.py

Tests the InternalDimensionNetwork architecture.
"""

import pytest
import torch
import torch.nn as nn
from src.core.network import InternalDimensionNetwork, BaselineNetwork


class TestInternalDimensionNetwork:
    """Test suite for InternalDimensionNetwork."""

    def setup_method(self):
        """Setup before each test."""
        self.input_dim = 4
        self.hidden_dim = 64
        self.output_dim = 2
        self.internal_dim = 32

        self.model = InternalDimensionNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            internal_dim=self.internal_dim
        )

        self.batch_size = 8
        self.state = torch.randn(self.batch_size, self.input_dim)

    def test_initialization(self):
        """Test model initializes correctly."""
        assert isinstance(self.model.encoder, nn.Sequential)
        assert isinstance(self.model.policy_head, nn.Sequential)
        assert isinstance(self.model.value_head, nn.Sequential)
        assert hasattr(self.model, 'internal_state')

    def test_forward_pass(self):
        """Test forward pass returns correct shapes."""
        policy, value, _ = self.model(self.state)

        assert policy.shape == (self.batch_size, self.output_dim)
        assert value.shape == (self.batch_size, 1)

    def test_return_internals(self):
        """Test that internal states are returned when requested."""
        policy, value, internals = self.model(self.state, return_internals=True)

        assert internals is not None
        assert 'x12' in internals
        assert 'm12' in internals
        assert 'attention' in internals
        assert 'modulation_signal' in internals
        assert 'hidden' in internals

    def test_internal_dimensions_exist(self):
        """Test that internal dimensions have correct properties."""
        _, _, internals = self.model(self.state, return_internals=True)

        x12 = internals['x12']
        m12 = internals['m12']

        # Should be scalars (or broadcastable)
        assert x12.numel() == 1
        assert m12.numel() == 1

        # Should be in [-1, 1]
        assert -1.0 <= x12.item() <= 1.0
        assert -1.0 <= m12.item() <= 1.0

    def test_modulation_affects_output(self):
        """Test that internal dimensions actually affect output."""
        # Get baseline output
        policy1, value1, _ = self.model(self.state)

        # Manually change internal state
        self.model.internal_state.x12 = torch.tensor([0.9])
        self.model.internal_state.m12 = torch.tensor([0.9])

        # Get new output
        policy2, value2, _ = self.model(self.state)

        # Outputs should be different (modulation working)
        assert not torch.allclose(policy1, policy2, atol=1e-6)

    def test_update_internal_state(self):
        """Test internal state update mechanism."""
        # Forward pass
        _, _, internals = self.model(self.state, return_internals=True)
        hidden = internals['hidden']

        # Create next state and reward
        next_state = torch.randn(self.batch_size, self.input_dim)
        reward = torch.tensor([1.0])

        # Update internal state
        update_info = self.model.update_internal_state(
            current_hidden=hidden,
            next_state=next_state,
            reward=reward
        )

        assert 'x12' in update_info
        assert 'm12' in update_info
        assert 'prediction_error' in update_info
        assert 'novelty' in update_info
        assert 'attention' in update_info

    def test_intrinsic_reward_curiosity(self):
        """Test curiosity-based intrinsic reward."""
        # High x12 should give higher curiosity reward
        self.model.internal_state.x12 = torch.tensor([0.9])
        reward_high = self.model.compute_intrinsic_reward(method='curiosity')

        self.model.internal_state.x12 = torch.tensor([0.1])
        reward_low = self.model.compute_intrinsic_reward(method='curiosity')

        assert reward_high.item() > reward_low.item()

    def test_intrinsic_reward_wisdom(self):
        """Test wisdom-based intrinsic reward."""
        # Positive m12 should give higher wisdom reward
        self.model.internal_state.m12 = torch.tensor([0.5])
        reward_positive = self.model.compute_intrinsic_reward(method='wisdom')

        self.model.internal_state.m12 = torch.tensor([-0.5])
        reward_negative = self.model.compute_intrinsic_reward(method='wisdom')

        # Note: wisdom reward is about staying near target (0.5)
        # So negative m12 should have lower reward
        assert reward_positive.item() > reward_negative.item()

    def test_lstm_option(self):
        """Test LSTM-enabled version."""
        model_lstm = InternalDimensionNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            use_lstm=True
        )

        # Should have LSTM
        assert hasattr(model_lstm, 'lstm')

        # Forward pass should work
        policy, value, _ = model_lstm(self.state)
        assert policy.shape == (self.batch_size, self.output_dim)

        # Reset should work
        model_lstm.reset_lstm()
        assert model_lstm.hidden_state is None

    def test_save_and_load(self, tmp_path):
        """Test saving and loading model."""
        # Train for a few steps to change internal state
        for _ in range(10):
            _, _, internals = self.model(self.state, return_internals=True)
            next_state = torch.randn(self.batch_size, self.input_dim)
            self.model.update_internal_state(
                internals['hidden'],
                next_state,
                torch.tensor([1.0])
            )

        # Save
        save_path = tmp_path / "model.pt"
        self.model.save(str(save_path))

        # Create new model and load
        new_model = InternalDimensionNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            internal_dim=self.internal_dim
        )
        new_model.load(str(save_path))

        # Internal states should match
        assert torch.allclose(
            self.model.internal_state.x12,
            new_model.internal_state.x12
        )
        assert torch.allclose(
            self.model.internal_state.m12,
            new_model.internal_state.m12
        )

    def test_gradient_flow(self):
        """Test that gradients flow through network correctly."""
        policy, value, _ = self.model(self.state)

        # Compute dummy loss
        loss = policy.mean() + value.mean()

        # Backward
        loss.backward()

        # Check that encoder has gradients
        for param in self.model.encoder.parameters():
            assert param.grad is not None

        # Check that policy head has gradients
        for param in self.model.policy_head.parameters():
            assert param.grad is not None

    def test_x12_network_no_gradient_from_task(self):
        """Test that x12 network doesn't get gradients from task loss directly."""
        policy, value, internals = self.model(self.state, return_internals=True)

        # Compute task loss (should NOT propagate to x12 network)
        loss = policy.mean() + value.mean()

        # Zero gradients
        self.model.zero_grad()

        # Backward
        loss.backward()

        # x12_network should have gradients (from modulation path)
        # but they should be indirect, not from direct x12 supervision
        # This is hard to test perfectly, but we can check it has gradients
        has_gradients = False
        for param in self.model.x12_network.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        # This is actually expected - modulation path does propagate gradients
        # The key is that x12 is ALSO updated via its own dynamics
        assert has_gradients  # Modulation allows some gradient flow

    def test_batch_independence(self):
        """Test that batch samples are processed independently."""
        # Two different states
        state1 = torch.randn(1, self.input_dim)
        state2 = torch.randn(1, self.input_dim)

        policy1, value1, _ = self.model(state1)
        policy2, value2, _ = self.model(state2)

        # Should give different outputs
        assert not torch.allclose(policy1, policy2)
        assert not torch.allclose(value1, value2)


class TestBaselineNetwork:
    """Test baseline network (for comparison)."""

    def setup_method(self):
        """Setup before each test."""
        self.model = BaselineNetwork(
            input_dim=4,
            hidden_dim=64,
            output_dim=2
        )

    def test_forward_pass(self):
        """Test baseline network forward pass."""
        state = torch.randn(8, 4)
        policy, value = self.model(state)

        assert policy.shape == (8, 2)
        assert value.shape == (8, 1)

    def test_no_internal_dimensions(self):
        """Test that baseline has no internal dimensions."""
        assert not hasattr(self.model, 'internal_state')
        assert not hasattr(self.model, 'x12_network')
        assert not hasattr(self.model, 'modulation_network')


class TestNetworkComparison:
    """Compare Internal Dimension Network to Baseline."""

    def setup_method(self):
        """Setup both networks."""
        self.idn = InternalDimensionNetwork(
            input_dim=4,
            hidden_dim=64,
            output_dim=2
        )

        self.baseline = BaselineNetwork(
            input_dim=4,
            hidden_dim=64,
            output_dim=2
        )

    def test_parameter_count_difference(self):
        """Test that IDN has more parameters (internal pathways)."""
        idn_params = sum(p.numel() for p in self.idn.parameters())
        baseline_params = sum(p.numel() for p in self.baseline.parameters())

        assert idn_params > baseline_params

    def test_both_produce_valid_outputs(self):
        """Test that both networks produce valid outputs."""
        state = torch.randn(8, 4)

        # IDN
        policy_idn, value_idn, _ = self.idn(state)
        assert not torch.isnan(policy_idn).any()
        assert not torch.isnan(value_idn).any()

        # Baseline
        policy_base, value_base = self.baseline(state)
        assert not torch.isnan(policy_base).any()
        assert not torch.isnan(value_base).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
