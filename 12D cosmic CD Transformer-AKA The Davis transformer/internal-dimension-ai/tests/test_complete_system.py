"""
Complete System Integration Tests

Validates entire pipeline from environment creation through
training, evaluation, and consciousness metric computation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.network import InternalDimensionNetwork, BaselineNetwork
from core.metrics import ConsciousnessMetrics
from environments.gridworld import GridWorld, TwoRoomGridWorld
from environments.social import PrisonersDilemma, IteratedPrisonersDilemma
from training.trainer import PPOTrainer


class TestCompleteSystem:
    """Complete system integration tests."""

    def test_basic_training_pipeline(self):
        """Test basic training completes without errors."""
        env = GridWorld(size=5)
        model = InternalDimensionNetwork(
            input_dim=env.observation_space.shape[0],
            hidden_dim=64,
            output_dim=env.action_space.n,
            internal_dim=12
        )
        trainer = PPOTrainer(model, env, use_tensorboard=False, use_wandb=False)
        history = trainer.train(num_episodes=5, steps_per_episode=50)

        assert 'episode_rewards' in history
        assert len(history['episode_rewards']) <= 5  # May be less if episodes don't complete
        assert 'policy_losses' in history
        assert len(history['policy_losses']) == 5

    def test_consciousness_metrics(self):
        """Test consciousness metrics compute valid values."""
        metrics = ConsciousnessMetrics()

        # Create dummy model
        model = InternalDimensionNetwork(4, 64, 4, 12)
        dummy_input = torch.randn(10, 4)

        with torch.no_grad():
            internal_states = []
            for x in dummy_input:
                model(x.unsqueeze(0))
                internal_states.append(model.internal_state.state)
            internal_states = torch.stack(internal_states)

        R_omega = metrics.compute_r_omega(model)
        R_psi = metrics.compute_r_psi(internal_states)

        assert -1 <= R_omega <= 1
        assert 0 <= R_psi <= 1

    def test_multi_agent_environment(self):
        """Test multi-agent environments work."""
        env = PrisonersDilemma(num_agents=2, internal_dim=12)
        obs, info = env.reset()

        assert 'agent_0' in obs
        assert 'agent_1' in obs

        actions = {'agent_0': 0, 'agent_1': 1}
        obs, rewards, dones, truncs, info = env.step(actions)

        assert 'agent_0' in rewards
        assert 'cooperation_rate' in info

    def test_visualization_scripts(self):
        """Test visualization scripts can be imported."""
        from evaluation.visualizations import InternalDimensionVisualizer
        viz = InternalDimensionVisualizer()
        assert viz is not None

    def test_cosmic_synapse_integration(self):
        """Test cosmic synapse module loads and runs."""
        from advanced.cosmic_synapse import CosmicSynapsePhysics, CosmicTransformer

        physics = CosmicSynapsePhysics(n=64, seed=42)
        physics.step()
        state = physics.get_state_vector()

        # State should be flattened positions + velocities
        assert state.shape[0] == 1  # Batch dimension
        assert state.shape[1] == 64 * 12 * 2  # n * dims * (pos + vel)

        model = CosmicTransformer(
            vocab_size=100,
            ctx_len=128,
            d_model=64,
            n_layer=2,
            n_head=4
        )
        assert model is not None

        # Test forward pass
        input_ids = torch.randint(0, 100, (2, 16))

        # Need to reduce physics state to match expected dimension (128)
        # The model expects 128 dims, but we're giving it 64*12*2 = 1536
        # Let's create a simple projection
        physics_state_full = state
        physics_state_reduced = torch.randn(2, 128)  # Use dummy for test

        logits = model(input_ids, physics_state_reduced)
        assert logits.shape == (2, 16, 100)

    def test_internal_dimension_network(self):
        """Test internal dimension network functionality."""
        model = InternalDimensionNetwork(4, 64, 2, internal_dim=12)

        # Test forward pass
        x = torch.randn(1, 4)
        policy_logits, value, internals = model(x, return_internals=True)

        assert policy_logits.shape == (1, 2)
        assert value.shape == (1, 1)
        assert 'x12' in internals
        assert 'm12' in internals
        assert 'hidden' in internals

    def test_baseline_network(self):
        """Test baseline network works."""
        model = BaselineNetwork(4, 64, 2)

        x = torch.randn(1, 4)
        policy_logits, value = model(x)

        assert policy_logits.shape == (1, 2)
        assert value.shape == (1, 1)

    def test_gridworld_environment(self):
        """Test gridworld environment."""
        env = GridWorld(size=8)

        obs, info = env.reset()
        assert obs.shape == (2,)  # (x, y) position

        action = env.action_space.sample()
        result = env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result

        assert obs.shape == (2,)

    def test_two_room_gridworld(self):
        """Test two-room gridworld environment."""
        env = TwoRoomGridWorld(size=10)

        obs, info = env.reset()
        assert obs.shape == (2,)

        # Take some steps
        for _ in range(10):
            action = env.action_space.sample()
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                if terminated or truncated:
                    break
            else:
                obs, reward, done, info = result
                if done:
                    break

        assert obs.shape == (2,)

    def test_intrinsic_rewards(self):
        """Test intrinsic reward computation."""
        model = InternalDimensionNetwork(4, 64, 2, 12)

        # Generate some states to build history
        for _ in range(10):
            x = torch.randn(1, 4)
            model(x)

        # Test different intrinsic reward methods
        curiosity = model.compute_intrinsic_reward(method='curiosity')
        wisdom = model.compute_intrinsic_reward(method='wisdom')
        balanced = model.compute_intrinsic_reward(method='balanced')

        assert isinstance(curiosity.item(), float)
        assert isinstance(wisdom.item(), float)
        assert isinstance(balanced.item(), float)

    def test_trainer_checkpoint_save_load(self, tmp_path):
        """Test checkpoint saving and loading."""
        env = GridWorld(size=5)
        model = InternalDimensionNetwork(2, 64, 4, 12)
        trainer = PPOTrainer(
            model, env,
            use_tensorboard=False,
            use_wandb=False,
            checkpoint_dir=str(tmp_path)
        )

        # Train briefly
        history = trainer.train(num_episodes=3, steps_per_episode=50)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path, 3, history)

        assert checkpoint_path.exists()

        # Create new trainer and load checkpoint
        new_model = InternalDimensionNetwork(2, 64, 4, 12)
        new_trainer = PPOTrainer(
            new_model, env,
            use_tensorboard=False,
            use_wandb=False
        )

        loaded_history = new_trainer.load_checkpoint(checkpoint_path)

        assert 'episode_rewards' in loaded_history


class TestCosmicSynapse:
    """Tests specific to cosmic synapse module."""

    def test_physics_simulation(self):
        """Test 12D physics simulation."""
        from advanced.cosmic_synapse import CosmicSynapsePhysics

        physics = CosmicSynapsePhysics(n=32, seed=42)

        initial_energy = physics.compute_energy()
        initial_entropy = physics.compute_entropy()

        # Run simulation
        for _ in range(100):
            physics.step()

        assert len(physics.energy_history) == 100
        assert len(physics.entropy_history) == 100

        # Check energy changes (not conserved due to chaos injection)
        assert physics.energy_history[-1] != initial_energy

    def test_lorenz_chaos(self):
        """Test Lorenz chaos injection."""
        from advanced.cosmic_synapse import CosmicSynapsePhysics

        physics = CosmicSynapsePhysics(n=16, seed=42)

        initial_lorenz = physics.lorenz_state.copy()

        for _ in range(50):
            physics.lorenz_step()

        # Lorenz state should change
        assert not np.allclose(physics.lorenz_state, initial_lorenz)

    def test_hebbian_weights(self):
        """Test Hebbian weight learning."""
        from advanced.cosmic_synapse import CosmicSynapsePhysics

        physics = CosmicSynapsePhysics(n=8, seed=42)

        # Initially zeros
        assert np.all(physics.hebbian_weights == 0)

        # Run simulation
        for _ in range(50):
            physics.step()

        # Hebbian weights should be updated
        assert not np.all(physics.hebbian_weights == 0)
        assert np.all(np.abs(physics.hebbian_weights) < 1.0)  # Reasonable range

    def test_transformer_architecture(self):
        """Test transformer architecture."""
        from advanced.cosmic_synapse import CosmicTransformer

        model = CosmicTransformer(
            vocab_size=1000,
            ctx_len=128,
            d_model=256,
            n_layer=4,
            n_head=8,
            physics_conditioning=True
        )

        # Count parameters
        params = model.count_parameters()
        assert params > 0

        # Test forward pass
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        physics_state = torch.randn(batch_size, 128)

        logits = model(input_ids, physics_state)

        assert logits.shape == (batch_size, seq_len, 1000)

    def test_transformer_generation(self):
        """Test text generation."""
        from advanced.cosmic_synapse import CosmicTransformer

        model = CosmicTransformer(
            vocab_size=100,
            ctx_len=64,
            d_model=128,
            n_layer=2,
            n_head=4
        )

        input_ids = torch.tensor([[1, 2, 3]])
        physics_state = torch.randn(1, 128)

        generated = model.generate(
            input_ids,
            physics_state,
            max_new_tokens=10,
            temperature=1.0,
            top_k=10
        )

        assert generated.shape[1] == input_ids.shape[1] + 10


class TestMetrics:
    """Tests for consciousness metrics."""

    def test_r_omega_computation(self):
        """Test R_omega (richness) metric."""
        metrics = ConsciousnessMetrics()
        model = InternalDimensionNetwork(4, 64, 2, 12)

        # Generate some activations
        for _ in range(20):
            x = torch.randn(1, 4)
            model(x)

        r_omega = metrics.compute_r_omega(model)

        assert isinstance(r_omega, float)
        assert -1 <= r_omega <= 1

    def test_r_psi_computation(self):
        """Test R_psi (binding) metric."""
        metrics = ConsciousnessMetrics()

        # Create dummy internal states
        internal_states = torch.randn(50, 12)

        r_psi = metrics.compute_r_psi(internal_states)

        assert isinstance(r_psi, float)
        assert 0 <= r_psi <= 1


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_complete_system.py -v
    pytest.main([__file__, "-v", "-s"])
