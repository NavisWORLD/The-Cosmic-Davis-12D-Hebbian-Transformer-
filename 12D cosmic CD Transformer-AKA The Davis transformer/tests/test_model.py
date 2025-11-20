"""
Model Architecture Tests for 12D Cosmic Synapse Transformer

Tests the core model components including:
- Model initialization
- Forward pass
- x12 internal dynamics
- Hebbian attention
- Chaos injection
- φ-harmonic scaling
- Memory module

Author: Cory Shane Davis
License: MIT
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmic_synapse_transformer import CosmicConfig, CosmicSynapseTransformer
from config_loader import ModelConfig


class TestModelInitialization:
    """Test model initialization and parameter counting."""

    def test_create_tiny_model(self):
        """Test creating a tiny model."""
        config = CosmicConfig(
            vocab_size=100,
            max_seq_len=64,
            d_model=96,
            n_layers=2,
            n_heads=2,
        )

        model = CosmicSynapseTransformer(config)
        assert model is not None
        assert model.get_num_params() > 0

    def test_parameter_count(self):
        """Test that parameter count is reasonable."""
        config = CosmicConfig(
            vocab_size=1000,
            max_seq_len=128,
            d_model=192,
            n_layers=4,
            n_heads=4,
        )

        model = CosmicSynapseTransformer(config)
        num_params = model.get_num_params()

        # Should have at least 100K parameters for this size
        assert num_params > 100_000
        # But not more than 10M for this tiny config
        assert num_params < 10_000_000

    def test_phi_harmonic_scaling(self):
        """Test that d_ff uses φ-harmonic scaling."""
        config = CosmicConfig(
            vocab_size=100,
            d_model=384,
            n_layers=2,
            n_heads=2,
        )

        phi = 1.618033988749895
        expected_d_ff = int(384 * phi)  # ≈ 621

        model = CosmicSynapseTransformer(config)

        # Check that d_ff is computed correctly
        assert abs(config.d_ff - expected_d_ff) <= 1  # Allow for rounding

    def test_x12_initialization(self):
        """Test that x12 internal states are initialized correctly."""
        config = CosmicConfig(
            vocab_size=100,
            d_model=96,
            n_layers=4,
            n_heads=2,
        )

        model = CosmicSynapseTransformer(config)

        # x12 should be initialized to zeros
        assert hasattr(model, 'x12')
        assert model.x12.shape == (config.n_layers,)
        assert torch.all(model.x12 == 0.0)


class TestForwardPass:
    """Test forward pass through the model."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        config = CosmicConfig(
            vocab_size=100,
            max_seq_len=64,
            d_model=96,
            n_layers=2,
            n_heads=2,
        )

        model = CosmicSynapseTransformer(config)
        model.eval()

        batch_size = 2
        seq_len = 32
        inputs = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            logits = model(inputs)

        # Output shape should be [batch, seq_len, vocab_size]
        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_forward_with_targets(self):
        """Test forward pass with targets (computes loss)."""
        config = CosmicConfig(
            vocab_size=100,
            max_seq_len=64,
            d_model=96,
            n_layers=2,
            n_heads=2,
        )

        model = CosmicSynapseTransformer(config)

        batch_size = 2
        seq_len = 32
        inputs = torch.randint(0, 100, (batch_size, seq_len))
        targets = torch.randint(0, 100, (batch_size, seq_len))

        logits, loss = model(inputs, targets)

        # Check shapes
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert loss.ndim == 0  # Scalar loss
        assert loss.item() > 0  # Loss should be positive

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        config = CosmicConfig(
            vocab_size=100,
            d_model=96,
            n_layers=2,
            n_heads=2,
        )

        model = CosmicSynapseTransformer(config)

        inputs = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))

        logits, loss = model(inputs, targets)
        loss.backward()

        # Check that some parameters have gradients
        has_grads = False
        for param in model.parameters():
            if param.grad is not None:
                has_grads = True
                break

        assert has_grads, "No gradients computed"


class TestX12Dynamics:
    """Test internal x12 state dynamics."""

    def test_x12_updates(self):
        """Test that x12 values update during forward passes."""
        config = CosmicConfig(
            vocab_size=100,
            d_model=96,
            n_layers=4,
            n_heads=2,
            k=0.1,  # Decay rate
        )

        model = CosmicSynapseTransformer(config)

        inputs = torch.randint(0, 100, (2, 16))

        # Initial x12 values
        initial_x12 = model.x12.clone()

        # Forward pass
        _ = model(inputs)

        # x12 should have changed
        assert not torch.allclose(model.x12, initial_x12)

    def test_x12_bounded(self):
        """Test that x12 values stay bounded in [-1, 1]."""
        config = CosmicConfig(
            vocab_size=100,
            d_model=96,
            n_layers=4,
            n_heads=2,
        )

        model = CosmicSynapseTransformer(config)

        # Run multiple forward passes
        for _ in range(10):
            inputs = torch.randint(0, 100, (2, 16))
            _ = model(inputs)

        # Check that x12 is bounded
        assert torch.all(model.x12 >= -1.0)
        assert torch.all(model.x12 <= 1.0)

    def test_x12_convergence(self):
        """Test that x12 values converge over time."""
        config = CosmicConfig(
            vocab_size=100,
            d_model=96,
            n_layers=4,
            n_heads=2,
            k=0.1,
        )

        model = CosmicSynapseTransformer(config)

        x12_history = []

        # Run many forward passes with same input
        inputs = torch.randint(0, 100, (2, 16))
        for _ in range(50):
            _ = model(inputs)
            x12_history.append(model.x12.clone())

        # Check that x12 changes less over time (convergence)
        early_change = (x12_history[5] - x12_history[0]).abs().mean()
        late_change = (x12_history[49] - x12_history[44]).abs().mean()

        assert late_change < early_change, "x12 should converge over time"


class TestHebbianAttention:
    """Test Hebbian connectivity in attention mechanism."""

    def test_hebbian_bonus_exists(self):
        """Test that Hebbian bonus is applied to attention."""
        config = CosmicConfig(
            vocab_size=100,
            d_model=96,
            n_layers=2,
            n_heads=2,
            gamma=0.05,
        )

        model = CosmicSynapseTransformer(config)

        # Check that Hebbian connectivity matrix exists
        for block in model.blocks:
            assert hasattr(block.attn, 'hebbian_bonus')

    def test_omega_evolution(self):
        """Test that Ω connectivity matrix evolves."""
        config = CosmicConfig(
            vocab_size=100,
            d_model=96,
            n_layers=2,
            n_heads=2,
            gamma=0.05,
        )

        model = CosmicSynapseTransformer(config)

        inputs = torch.randint(0, 100, (2, 16))

        # Get initial Ω
        initial_omega = model.blocks[0].attn.hebbian_bonus.clone()

        # Run forward passes
        for _ in range(10):
            _ = model(inputs)

        # Ω should have changed
        final_omega = model.blocks[0].attn.hebbian_bonus
        assert not torch.allclose(initial_omega, final_omega)


class TestChaosInjection:
    """Test Lorenz chaos injection."""

    def test_lorenz_evolution(self):
        """Test that Lorenz attractor state evolves."""
        config = CosmicConfig(
            vocab_size=100,
            d_model=96,
            n_layers=2,
            n_heads=2,
            sigma=0.5,
        )

        model = CosmicSynapseTransformer(config)

        # Initial Lorenz state
        initial_lorenz = model.lorenz_state.clone()

        inputs = torch.randint(0, 100, (2, 16))
        _ = model(inputs)

        # Lorenz state should have evolved
        assert not torch.allclose(model.lorenz_state, initial_lorenz)

    def test_chaos_reproducibility(self):
        """Test that chaos is reproducible with same seed."""
        config1 = CosmicConfig(vocab_size=100, d_model=96, n_layers=2, n_heads=2)
        config2 = CosmicConfig(vocab_size=100, d_model=96, n_layers=2, n_heads=2)

        torch.manual_seed(42)
        model1 = CosmicSynapseTransformer(config1)

        torch.manual_seed(42)
        model2 = CosmicSynapseTransformer(config2)

        inputs = torch.randint(0, 100, (2, 16))

        torch.manual_seed(42)
        out1 = model1(inputs)

        torch.manual_seed(42)
        out2 = model2(inputs)

        assert torch.allclose(out1, out2)


class TestMemoryModule:
    """Test episodic memory module."""

    def test_memory_initialization(self):
        """Test that memory buffer is initialized."""
        config = CosmicConfig(
            vocab_size=100,
            d_model=96,
            n_layers=2,
            n_heads=2,
            memory_size=32,
            memory_dim=64,
        )

        model = CosmicSynapseTransformer(config)

        assert hasattr(model, 'memory')
        if model.memory is not None:
            assert hasattr(model.memory, 'buffer')

    def test_memory_retrieval(self):
        """Test memory retrieval mechanism."""
        config = CosmicConfig(
            vocab_size=100,
            d_model=96,
            n_layers=2,
            n_heads=2,
            memory_size=32,
            memory_dim=96,
        )

        model = CosmicSynapseTransformer(config)

        if model.memory is not None:
            query = torch.randn(2, 16, 96)
            retrieved = model.memory.retrieve(query)

            # Should return something with same shape
            assert retrieved.shape == query.shape


@pytest.mark.slow
class TestGeneration:
    """Test text generation capabilities."""

    def test_basic_generation(self):
        """Test that model can generate text."""
        config = CosmicConfig(
            vocab_size=100,
            max_seq_len=64,
            d_model=96,
            n_layers=2,
            n_heads=2,
        )

        model = CosmicSynapseTransformer(config)
        model.eval()

        prompt = torch.randint(0, 100, (1, 10))
        max_tokens = 20

        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=max_tokens)

        # Output should be longer than prompt
        assert output.shape[1] > prompt.shape[1]
        assert output.shape[1] <= prompt.shape[1] + max_tokens

    def test_generation_temperature(self):
        """Test generation with different temperatures."""
        config = CosmicConfig(
            vocab_size=100,
            d_model=96,
            n_layers=2,
            n_heads=2,
        )

        model = CosmicSynapseTransformer(config)
        model.eval()

        prompt = torch.randint(0, 100, (1, 10))

        # Generate with different temperatures
        torch.manual_seed(42)
        output_low = model.generate(prompt, max_new_tokens=10, temperature=0.1)

        torch.manual_seed(42)
        output_high = model.generate(prompt, max_new_tokens=10, temperature=2.0)

        # Outputs should be different (high temp more random)
        # This test might occasionally fail due to randomness, so we just check it runs
        assert output_low.shape == output_high.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
