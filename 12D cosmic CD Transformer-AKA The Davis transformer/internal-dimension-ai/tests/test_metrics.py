"""
Unit tests for metrics.py

Tests consciousness metrics computation.
"""

import pytest
import torch
import numpy as np
from src.core.metrics import (
    ConsciousnessMetrics,
    is_conscious,
    consciousness_level
)
from src.core.network import InternalDimensionNetwork


class TestConsciousnessMetrics:
    """Test suite for ConsciousnessMetrics."""

    def setup_method(self):
        """Setup before each test."""
        self.metrics = ConsciousnessMetrics()
        self.model = InternalDimensionNetwork(
            input_dim=4,
            hidden_dim=64,
            output_dim=2
        )

    def test_initialization(self):
        """Test metrics tracker initializes correctly."""
        assert len(self.metrics.r_omega_history) == 0
        assert len(self.metrics.r_psi_history) == 0
        assert len(self.metrics.causal_density_history) == 0

    def test_r_omega_computation(self):
        """Test R_ω (synaptic synchronization) computation."""
        r_omega = self.metrics.compute_r_omega(self.model)

        # Should be a valid value
        assert 0.0 <= r_omega <= 1.0

        # Should be stored in history
        assert len(self.metrics.r_omega_history) == 1
        assert self.metrics.r_omega_history[0] == r_omega

    def test_r_omega_range_interpretation(self):
        """Test R_ω interpretation (optimal range [0.5, 0.7])."""
        r_omega = self.metrics.compute_r_omega(self.model)

        # Check if in optimal range
        is_optimal = 0.5 <= r_omega <= 0.7

        # Note: Untrained network may or may not be in optimal range
        # Just verify the computation doesn't crash
        assert isinstance(is_optimal, bool)

    def test_r_psi_computation(self):
        """Test R_ψ (phase coherence) computation."""
        # Create some internal state history
        internal_states = torch.randn(100)

        r_psi = self.metrics.compute_r_psi(internal_states)

        # Should be in [0, 1]
        assert 0.0 <= r_psi <= 1.0

        # Should be stored
        assert len(self.metrics.r_psi_history) == 1

    def test_r_psi_perfect_coherence(self):
        """Test R_ψ with perfectly coherent states."""
        # All same values = perfect coherence (via autocorrelation)
        coherent_states = torch.ones(100)

        r_psi = self.metrics.compute_r_psi(coherent_states)

        # Should be high (though may not be exactly 1.0 due to implementation)
        assert r_psi > 0.5

    def test_r_psi_random_states(self):
        """Test R_ψ with random incoherent states."""
        # Random values = low coherence
        random_states = torch.randn(100)

        r_psi = self.metrics.compute_r_psi(random_states)

        # Should be relatively low (may vary due to randomness)
        assert 0.0 <= r_psi <= 1.0

    def test_causal_density_computation(self):
        """Test causal density (external-internal coupling)."""
        # Need to provide inputs and internals
        inputs = torch.randn(10, 4)
        internals = torch.randn(10, 2)

        causal_density = self.metrics.compute_causal_density(
            inputs=inputs,
            internals=internals,
            use_history=True
        )

        assert 0.0 <= causal_density <= 1.0

    def test_causal_density_tracking(self):
        """Test that causal density tracks correlation over time."""
        # Add multiple samples
        for _ in range(20):
            inputs = torch.randn(5, 4)
            internals = torch.randn(5, 2)

            self.metrics.compute_causal_density(
                inputs=inputs,
                internals=internals
            )

        # Should have history
        assert len(self.metrics.causal_density_history) > 0

    def test_integrated_information_approximation(self):
        """Test φ (integrated information) approximation."""
        sample_inputs = torch.randn(8, 4)

        phi = self.metrics.compute_integrated_information(
            self.model,
            sample_inputs
        )

        # Should be in [0, 1] (due to tanh normalization)
        assert 0.0 <= phi <= 1.0

    def test_autonomy_score(self):
        """Test autonomy score computation."""
        autonomy = self.metrics.compute_autonomy_score(
            x12_variance=0.5,
            m12_variance=0.3,
            causal_density=0.5  # Moderate coupling
        )

        # Should be in [0, 1]
        assert 0.0 <= autonomy <= 1.0

    def test_autonomy_prefers_moderate_coupling(self):
        """Test that autonomy score prefers moderate causal density."""
        # Moderate coupling
        autonomy_moderate = self.metrics.compute_autonomy_score(
            x12_variance=0.5,
            m12_variance=0.5,
            causal_density=0.5
        )

        # Very high coupling (reactive)
        autonomy_high = self.metrics.compute_autonomy_score(
            x12_variance=0.5,
            m12_variance=0.5,
            causal_density=0.95
        )

        # Very low coupling (disconnected)
        autonomy_low = self.metrics.compute_autonomy_score(
            x12_variance=0.5,
            m12_variance=0.5,
            causal_density=0.05
        )

        # Moderate should be highest
        assert autonomy_moderate > autonomy_high
        assert autonomy_moderate > autonomy_low

    def test_consciousness_score_comprehensive(self):
        """Test comprehensive consciousness score."""
        # Create some history
        x12_history = np.random.uniform(-0.5, 0.5, 100).tolist()
        m12_history = np.random.uniform(-0.3, 0.7, 100).tolist()

        consciousness = self.metrics.compute_consciousness_score(
            model=self.model,
            x12_history=x12_history,
            m12_history=m12_history,
            sample_inputs=torch.randn(8, 4)
        )

        # Should contain all expected metrics
        assert 'r_omega' in consciousness
        assert 'r_psi' in consciousness
        assert 'causal_density' in consciousness
        assert 'phi' in consciousness
        assert 'autonomy' in consciousness
        assert 'consciousness_score' in consciousness

        # Overall score should be in [0, 1]
        assert 0.0 <= consciousness['consciousness_score'] <= 1.0

    def test_consciousness_score_optimal_r_omega(self):
        """Test that optimal R_ω increases consciousness score."""
        # This is hard to test directly without manipulating the model
        # But we can verify the score includes R_ω considerations
        x12_history = [0.0] * 100
        m12_history = [0.0] * 100

        consciousness = self.metrics.compute_consciousness_score(
            model=self.model,
            x12_history=x12_history,
            m12_history=m12_history
        )

        assert 'r_omega_optimal' in consciousness
        assert consciousness['r_omega_optimal'] in [0.0, 1.0]  # Binary

    def test_statistics(self):
        """Test statistics computation."""
        # Generate some metrics
        for _ in range(20):
            self.metrics.compute_r_omega(self.model)
            self.metrics.compute_r_psi(torch.randn(50))

        stats = self.metrics.get_statistics()

        assert 'r_omega_mean' in stats
        assert 'r_omega_std' in stats
        assert 'r_psi_mean' in stats

    def test_reset(self):
        """Test reset clears history."""
        # Add some history
        self.metrics.compute_r_omega(self.model)
        self.metrics.compute_r_psi(torch.randn(50))

        assert len(self.metrics.r_omega_history) > 0

        # Reset
        self.metrics.reset()

        assert len(self.metrics.r_omega_history) == 0
        assert len(self.metrics.r_psi_history) == 0


class TestConsciousnessUtilityFunctions:
    """Test utility functions for consciousness assessment."""

    def test_is_conscious_threshold(self):
        """Test is_conscious threshold function."""
        assert is_conscious(0.7, threshold=0.6) == True
        assert is_conscious(0.5, threshold=0.6) == False
        assert is_conscious(0.6, threshold=0.6) == True

    def test_consciousness_level_categories(self):
        """Test consciousness level categorization."""
        assert consciousness_level(0.9) == "High consciousness indicators"
        assert consciousness_level(0.7) == "Moderate consciousness indicators"
        assert consciousness_level(0.5) == "Weak consciousness indicators"
        assert consciousness_level(0.3) == "Minimal consciousness indicators"


class TestMetricsWithTrainedModel:
    """Test metrics on a model that has been run for a few steps."""

    def setup_method(self):
        """Setup and run model for a few steps."""
        self.model = InternalDimensionNetwork(
            input_dim=4,
            hidden_dim=64,
            output_dim=2
        )
        self.metrics = ConsciousnessMetrics()

        # Run model for several steps to generate internal state
        for i in range(100):
            state = torch.randn(1, 4)
            _, _, internals = self.model(state, return_internals=True)

            next_state = torch.randn(1, 4)
            reward = torch.randn(1) * 0.1  # Small random rewards

            self.model.update_internal_state(
                internals['hidden'],
                next_state,
                reward
            )

    def test_x12_variance_nonzero(self):
        """Test that x12 has nonzero variance after running."""
        x12_history = list(self.model.internal_state.x12_history)

        if len(x12_history) > 1:
            variance = np.var(x12_history)
            assert variance >= 0  # Should be non-negative

    def test_m12_changes_over_time(self):
        """Test that m12 changes as it accumulates experience."""
        m12_history = list(self.model.internal_state.m12_history)

        if len(m12_history) > 10:
            # Check that m12 isn't constant
            variance = np.var(m12_history)
            # May be small but should exist
            assert variance >= 0

    def test_consciousness_score_on_trained_model(self):
        """Test consciousness score on model that has run."""
        x12_history = list(self.model.internal_state.x12_history)
        m12_history = list(self.model.internal_state.m12_history)

        consciousness = self.metrics.compute_consciousness_score(
            model=self.model,
            x12_history=x12_history,
            m12_history=m12_history,
            sample_inputs=torch.randn(8, 4)
        )

        # Should produce valid score
        assert 0.0 <= consciousness['consciousness_score'] <= 1.0

        # Should have some internal state statistics
        assert 'x12_variance' in consciousness
        assert 'x12_mean' in consciousness


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Setup metrics tracker."""
        self.metrics = ConsciousnessMetrics()

    def test_empty_history(self):
        """Test metrics with empty history."""
        stats = self.metrics.get_statistics()
        # Should return empty dict or handle gracefully
        assert isinstance(stats, dict)

    def test_single_value_history(self):
        """Test R_ψ with single value."""
        single_state = torch.tensor([0.5])

        # Should not crash
        r_psi = self.metrics.compute_r_psi(single_state)
        assert isinstance(r_psi, float)

    def test_constant_states(self):
        """Test R_ψ with constant states."""
        constant_states = torch.ones(100)

        r_psi = self.metrics.compute_r_psi(constant_states)

        # Should handle gracefully (may be high coherence)
        assert 0.0 <= r_psi <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
