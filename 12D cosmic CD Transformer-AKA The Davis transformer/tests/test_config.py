"""
Configuration System Tests for 12D Cosmic Synapse Transformer

Tests configuration loading, validation, and management.

Author: Cory Shane Davis
License: MIT
"""

import pytest
import sys
from pathlib import Path
import tempfile
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    Config,
    load_config,
    load_config_from_dict,
    create_default_config,
)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_basic_creation(self):
        """Test creating a basic model config."""
        config = ModelConfig(
            vocab_size=1000,
            max_seq_len=128,
            d_model=192,
            n_layers=4,
            n_heads=4,
        )

        assert config.vocab_size == 1000
        assert config.d_model == 192
        assert config.n_layers == 4

    def test_phi_scaling(self):
        """Test that d_ff is computed using φ when not specified."""
        config = ModelConfig(
            vocab_size=1000,
            d_model=384,
            n_layers=2,
            n_heads=2,
        )

        # d_ff should be approximately d_model * 1.618
        phi = 1.618033988749895
        expected = int(384 * phi)
        assert abs(config.d_ff - expected) <= 1

    def test_validation(self):
        """Test config validation."""
        # Invalid: negative vocab_size
        with pytest.raises(AssertionError):
            ModelConfig(vocab_size=-1, d_model=96, n_layers=2, n_heads=2)

        # Invalid: d_model not divisible by n_heads
        with pytest.raises(AssertionError):
            ModelConfig(vocab_size=100, d_model=97, n_layers=2, n_heads=4)

        # Invalid: dropout out of range
        with pytest.raises(AssertionError):
            ModelConfig(vocab_size=100, d_model=96, n_layers=2, n_heads=2, dropout=1.5)


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_basic_creation(self):
        """Test creating a training config."""
        config = TrainingConfig(
            batch_size=8,
            max_iters=1000,
            learning_rate=0.001,
        )

        assert config.batch_size == 8
        assert config.max_iters == 1000
        assert config.learning_rate == 0.001

    def test_validation(self):
        """Test training config validation."""
        # Invalid: negative batch size
        with pytest.raises(AssertionError):
            TrainingConfig(batch_size=-1)

        # Invalid: invalid device
        with pytest.raises(AssertionError):
            TrainingConfig(device='gpu')  # Should be 'cuda'

        # Invalid: invalid dtype
        with pytest.raises(AssertionError):
            TrainingConfig(dtype='float64')


class TestDataConfig:
    """Test DataConfig dataclass."""

    def test_path_construction(self):
        """Test path construction methods."""
        config = DataConfig(
            data_dir='mydata',
            train_bin='train.bin',
            val_bin='val.bin',
        )

        train_path = config.get_train_path()
        val_path = config.get_val_path()

        assert str(train_path) == 'mydata/train.bin'
        assert str(val_path) == 'mydata/val.bin'


class TestConfigLoading:
    """Test loading configs from files."""

    def test_load_from_yaml(self):
        """Test loading config from YAML file."""
        config_dict = {
            'model': {
                'vocab_size': 1000,
                'max_seq_len': 128,
                'd_model': 192,
                'n_layers': 4,
                'n_heads': 4,
            },
            'training': {
                'batch_size': 4,
                'max_iters': 1000,
                'learning_rate': 0.0003,
            },
            'data': {
                'data_dir': 'data',
                'train_bin': 'train.bin',
                'val_bin': 'val.bin',
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config.model.vocab_size == 1000
            assert config.training.batch_size == 4
            assert config.data.data_dir == 'data'
        finally:
            Path(temp_path).unlink()

    def test_load_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {
            'model': {
                'vocab_size': 1000,
                'd_model': 192,
                'n_layers': 4,
                'n_heads': 4,
            },
            'training': {
                'batch_size': 4,
                'max_iters': 1000,
            },
            'data': {
                'data_dir': 'data',
            },
        }

        config = load_config_from_dict(config_dict)

        assert config.model.vocab_size == 1000
        assert config.training.batch_size == 4

    def test_load_missing_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent.yaml')

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        config_dict = {
            'model': {
                'vocab_size': 1000,
                'd_model': 192,
                'n_layers': 4,
                'n_heads': 4,
            },
            # Missing 'training' and 'data' sections
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()


class TestDefaultConfigs:
    """Test creating default configurations."""

    def test_create_tiny_config(self):
        """Test creating tiny default config."""
        config = create_default_config('tiny', 'cpu')

        assert config.model.vocab_size == 1000
        assert config.model.d_model == 192
        assert config.training.device == 'cpu'

    def test_create_small_config(self):
        """Test creating small default config."""
        config = create_default_config('small', 'cuda')

        assert config.model.vocab_size == 5000
        assert config.model.d_model == 384
        assert config.training.device == 'cuda'

    def test_create_medium_config(self):
        """Test creating medium default config."""
        config = create_default_config('medium', 'cuda')

        assert config.model.vocab_size == 50257
        assert config.model.d_model == 768

    def test_create_large_config(self):
        """Test creating large default config."""
        config = create_default_config('large', 'cuda')

        assert config.model.vocab_size == 50257
        assert config.model.d_model == 1024

    def test_invalid_model_size(self):
        """Test invalid model size raises error."""
        with pytest.raises(ValueError):
            create_default_config('huge', 'cpu')


class TestConfigSerialization:
    """Test config serialization."""

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = create_default_config('tiny', 'cpu')
        config_dict = config.to_dict()

        assert 'model' in config_dict
        assert 'training' in config_dict
        assert 'data' in config_dict

        assert config_dict['model']['vocab_size'] == 1000

    def test_save_and_load(self):
        """Test saving and loading config."""
        config = create_default_config('tiny', 'cpu')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            config.save(temp_path)

            # Load it back
            loaded_config = load_config(temp_path)

            # Should match
            assert loaded_config.model.vocab_size == config.model.vocab_size
            assert loaded_config.model.d_model == config.model.d_model
            assert loaded_config.training.batch_size == config.training.batch_size
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
