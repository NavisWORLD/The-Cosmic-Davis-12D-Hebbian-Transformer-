"""
Unit tests for internal_dimensions.py

Tests the core mathematical operations for x₁₂ and m₁₂ computation.
"""

import pytest
import torch
import numpy as np
from src.core.internal_dimensions import (
    InternalDimensionState,
    compute_importance,
    create_internal_state
)


class TestInternalDimensionState:
    """Test suite for InternalDimensionState class."""

    def setup_method(self):
        """Setup before each test."""
        self.device = torch.device('cpu')
        self.internal_state = InternalDimensionState(device=self.device)

    def test_initialization(self):
        """Test that internal state initializes correctly."""
        assert self.internal_state.x12.shape == (1,)
        assert self.internal_state.m12.shape == (1,)
        assert self.internal_state.x12.item() == 0.0
        assert self.internal_state.m12.item() == 0.0

    def test_x12_bounded(self):
        """Test that x₁₂ stays in [-1, 1] range."""
        # Create large prediction error
        large_error = torch.tensor(100.0)

        x12 = self.internal_state.compute_x12(
            prediction_error=large_error,
            novelty=torch.tensor(10.0),
            attention=torch.tensor(5.0)
        )

        assert -1.0 <= x12.item() <= 1.0

    def test_m12_bounded(self):
        """Test that m₁₂ stays in [-1, 1] range."""
        # Integrate many times with large x12
        for _ in range(1000):
            self.internal_state.update_m12(
                x12=torch.tensor(0.9),
                importance=torch.tensor(1.0)
            )

        assert -1.0 <= self.internal_state.m12.item() <= 1.0

    def test_x12_increases_with_surprise(self):
        """Test that x₁₂ increases with prediction error."""
        # Low surprise
        x12_low = self.internal_state.compute_x12(
            prediction_error=torch.tensor(0.01),
            novelty=torch.tensor(0.0),
            attention=torch.tensor(0.0)
        )

        # Reset
        self.internal_state.reset()

        # High surprise
        x12_high = self.internal_state.compute_x12(
            prediction_error=torch.tensor(10.0),
            novelty=torch.tensor(0.0),
            attention=torch.tensor(0.0)
        )

        assert x12_high.item() > x12_low.item()

    def test_m12_accumulates_x12(self):
        """Test that m₁₂ accumulates x₁₂ over time."""
        initial_m12 = self.internal_state.m12.item()

        # Positive x12 several times
        for _ in range(10):
            self.internal_state.compute_x12(
                prediction_error=torch.tensor(1.0),
                novelty=torch.tensor(0.0),
                attention=torch.tensor(0.0)
            )
            self.internal_state.update_m12(
                importance=torch.tensor(1.0)
            )

        final_m12 = self.internal_state.m12.item()

        assert final_m12 > initial_m12

    def test_prediction_error_computation(self):
        """Test prediction error calculation."""
        predicted = torch.tensor([1.0, 2.0, 3.0])
        actual = torch.tensor([1.0, 2.0, 3.0])

        error = self.internal_state.compute_prediction_error(predicted, actual)

        assert error.item() == pytest.approx(0.0)

        # Different values
        predicted = torch.tensor([0.0, 0.0, 0.0])
        actual = torch.tensor([1.0, 1.0, 1.0])

        error = self.internal_state.compute_prediction_error(predicted, actual)

        assert error.item() > 0.0

    def test_novelty_decreases_with_familiarity(self):
        """Test that novelty decreases as state becomes familiar."""
        state = torch.tensor([1.0, 2.0, 3.0])

        # First visit - should be novel
        novelty1 = self.internal_state.compute_novelty(state)

        # Visit same state many times
        for _ in range(100):
            self.internal_state.compute_novelty(state)

        # Last visit - should be less novel
        novelty_last = self.internal_state.compute_novelty(state)

        assert novelty1.item() > novelty_last.item()

    def test_attention_computation(self):
        """Test attention signal computation."""
        # High variance hidden state = high attention
        high_variance_state = torch.randn(128) * 10

        attention_high = self.internal_state.compute_attention(
            high_variance_state,
            method='variance'
        )

        # Low variance hidden state = low attention
        low_variance_state = torch.ones(128) * 0.1

        attention_low = self.internal_state.compute_attention(
            low_variance_state,
            method='variance'
        )

        assert attention_high.item() > attention_low.item()

    def test_history_tracking(self):
        """Test that history is properly maintained."""
        # Generate some x12 and m12 values
        for i in range(10):
            self.internal_state.compute_x12(
                prediction_error=torch.tensor(float(i) * 0.1),
                novelty=torch.tensor(0.0),
                attention=torch.tensor(0.0)
            )
            self.internal_state.update_m12(
                importance=torch.tensor(1.0)
            )

        assert len(self.internal_state.x12_history) == 10
        assert len(self.internal_state.m12_history) == 10

    def test_statistics(self):
        """Test statistics computation."""
        # Generate varied x12 values
        for i in range(50):
            self.internal_state.compute_x12(
                prediction_error=torch.tensor(float(i % 10) * 0.1),
                novelty=torch.tensor(0.0),
                attention=torch.tensor(0.0)
            )

        stats = self.internal_state.get_statistics()

        assert 'x12_mean' in stats
        assert 'x12_std' in stats
        assert 'x12_min' in stats
        assert 'x12_max' in stats

    def test_reset(self):
        """Test reset functionality."""
        # Set some values
        self.internal_state.compute_x12(
            prediction_error=torch.tensor(5.0),
            novelty=torch.tensor(1.0),
            attention=torch.tensor(1.0)
        )
        self.internal_state.update_m12(importance=torch.tensor(1.0))

        # Reset x12 only
        self.internal_state.reset(reset_m12=False)

        assert self.internal_state.x12.item() == 0.0
        assert len(self.internal_state.x12_history) == 0
        assert self.internal_state.m12.item() != 0.0  # Should preserve m12

        # Reset everything
        self.internal_state.reset(reset_m12=True)

        assert self.internal_state.m12.item() == 0.0
        assert len(self.internal_state.m12_history) == 0

    def test_homeostatic_regulation(self):
        """Test that m₁₂ returns to baseline without input."""
        # Set m12 away from baseline
        for _ in range(50):
            self.internal_state.compute_x12(
                prediction_error=torch.tensor(1.0),
                novelty=torch.tensor(0.0),
                attention=torch.tensor(0.0)
            )
            self.internal_state.update_m12(importance=torch.tensor(1.0))

        m12_before = self.internal_state.m12.item()

        # Update with zero x12 many times (homeostatic regulation should kick in)
        self.internal_state.reset()  # Reset x12 to 0
        for _ in range(1000):
            self.internal_state.update_m12(
                x12=torch.tensor(0.0),
                importance=torch.tensor(0.0)
            )

        m12_after = self.internal_state.m12.item()

        # Should move toward baseline (0.0)
        assert abs(m12_after) < abs(m12_before)


class TestImportanceFunction:
    """Test importance weight computation."""

    def test_absolute_importance(self):
        """Test absolute value method."""
        reward = torch.tensor(-5.0)
        importance = compute_importance(reward, method='absolute')
        assert importance.item() == 5.0

    def test_signed_importance(self):
        """Test signed method."""
        reward = torch.tensor(-5.0)
        importance = compute_importance(reward, method='signed')
        assert importance.item() == -5.0

    def test_squared_importance(self):
        """Test squared method."""
        reward = torch.tensor(-5.0)
        importance = compute_importance(reward, method='squared')
        assert importance.item() == 25.0


class TestFactoryFunction:
    """Test factory function for creating internal states."""

    def test_create_internal_state(self):
        """Test factory function."""
        state = create_internal_state(
            alpha=2.0,
            beta=0.8,
            eta=0.05
        )

        assert state.alpha == 2.0
        assert state.beta == 0.8
        assert state.eta == 0.05

    def test_device_auto_detection(self):
        """Test automatic device selection."""
        state = create_internal_state()
        assert state.device.type in ['cpu', 'cuda']


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Setup before each test."""
        self.internal_state = InternalDimensionState()

    def test_nan_inputs(self):
        """Test handling of NaN inputs."""
        # Should not crash
        try:
            self.internal_state.compute_x12(
                prediction_error=torch.tensor(float('nan')),
                novelty=torch.tensor(0.0),
                attention=torch.tensor(0.0)
            )
        except:
            pytest.fail("Should handle NaN gracefully")

    def test_large_values(self):
        """Test with very large values."""
        x12 = self.internal_state.compute_x12(
            prediction_error=torch.tensor(1e10),
            novelty=torch.tensor(1e10),
            attention=torch.tensor(1e10)
        )

        # Should still be bounded
        assert -1.0 <= x12.item() <= 1.0

    def test_negative_prediction_error(self):
        """Test that prediction error is always positive."""
        predicted = torch.tensor([5.0])
        actual = torch.tensor([3.0])

        error = self.internal_state.compute_prediction_error(predicted, actual)

        assert error.item() >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
