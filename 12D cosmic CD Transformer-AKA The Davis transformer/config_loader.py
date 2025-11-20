"""
Configuration Loader for 12D Cosmic Synapse Transformer

This module provides utilities to load and validate YAML configuration files,
converting them into structured config objects for the model and training.

Author: Cory Shane Davis
License: MIT
"""

import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import math


@dataclass
class ModelConfig:
    """Configuration for the Cosmic Synapse Transformer model."""

    # Architecture
    vocab_size: int = 1000
    max_seq_len: int = 128
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 4
    d_ff: Optional[int] = None  # If None, computed as d_model * φ
    dropout: float = 0.1

    # 12D Cosmic Synapse Parameters
    k: float = 0.1      # Internal state decay rate
    gamma: float = 0.05  # Hebbian learning rate
    sigma: float = 0.5   # Chaos injection strength
    beta: float = 0.2    # Lorenz attractor parameter

    # Memory parameters
    memory_size: int = 32
    memory_dim: int = 64

    def __post_init__(self):
        """Validate and compute derived parameters."""
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895

        # Compute d_ff using φ-harmonic scaling if not specified
        if self.d_ff is None:
            self.d_ff = int(self.d_model * phi)

        # Validation
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert 0 <= self.dropout <= 1, "dropout must be in [0, 1]"

        # Cosmic Synapse parameter validation
        assert self.k > 0, "k (decay rate) must be positive"
        assert self.gamma > 0, "gamma (Hebbian rate) must be positive"
        assert self.sigma >= 0, "sigma (chaos strength) must be non-negative"
        assert self.beta > 0, "beta (Lorenz parameter) must be positive"

        # Memory validation
        assert self.memory_size > 0, "memory_size must be positive"
        assert self.memory_dim > 0, "memory_dim must be positive"


@dataclass
class TrainingConfig:
    """Configuration for training the model."""

    # Batch and iterations
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 10
    log_interval: int = 10

    # Optimization
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Learning rate schedule
    warmup_iters: int = 100
    lr_decay_iters: int = 1000
    min_lr: float = 0.00003

    # Device and precision
    device: str = 'cpu'
    dtype: str = 'float32'
    compile: bool = False

    # Distributed training
    ddp: bool = False
    ddp_backend: str = 'nccl'

    # Checkpointing
    out_dir: str = 'checkpoints/model'
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'  # 'scratch' or 'resume' or checkpoint path

    def __post_init__(self):
        """Validate training parameters."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.max_iters > 0, "max_iters must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 <= self.weight_decay <= 1, "weight_decay must be in [0, 1]"
        assert 0 < self.beta1 < 1, "beta1 must be in (0, 1)"
        assert 0 < self.beta2 < 1, "beta2 must be in (0, 1)"
        assert self.grad_clip > 0, "grad_clip must be positive"
        assert self.device in ['cpu', 'cuda', 'mps'], f"Invalid device: {self.device}"
        assert self.dtype in ['float32', 'float16', 'bfloat16'], f"Invalid dtype: {self.dtype}"


@dataclass
class DataConfig:
    """Configuration for data loading."""

    data_dir: str = 'data'
    train_bin: str = 'train.bin'
    val_bin: str = 'val.bin'

    def get_train_path(self) -> Path:
        """Get full path to training data."""
        return Path(self.data_dir) / self.train_bin

    def get_val_path(self) -> Path:
        """Get full path to validation data."""
        return Path(self.data_dir) / self.val_bin


@dataclass
class Config:
    """Complete configuration combining all sub-configs."""

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
        }

    def save(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config object with validated parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Validate structure
    required_keys = {'model', 'training', 'data'}
    if not required_keys.issubset(config_dict.keys()):
        missing = required_keys - config_dict.keys()
        raise ValueError(f"Config missing required sections: {missing}")

    # Create config objects
    try:
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        data_config = DataConfig(**config_dict['data'])

        config = Config(
            model=model_config,
            training=training_config,
            data=data_config
        )

        return config

    except TypeError as e:
        raise ValueError(f"Invalid config parameters: {e}")


def load_config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """
    Load configuration from dictionary.

    Args:
        config_dict: Dictionary with config parameters

    Returns:
        Config object
    """
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    data_config = DataConfig(**config_dict.get('data', {}))

    return Config(
        model=model_config,
        training=training_config,
        data=data_config
    )


def create_default_config(
    model_size: str = 'tiny',
    device: str = 'cpu'
) -> Config:
    """
    Create a default configuration for quick testing.

    Args:
        model_size: One of 'tiny', 'small', 'medium', 'large'
        device: Device to use ('cpu', 'cuda', 'mps')

    Returns:
        Config object
    """
    configs = {
        'tiny': {
            'vocab_size': 1000,
            'max_seq_len': 128,
            'd_model': 192,
            'n_layers': 4,
            'n_heads': 4,
            'batch_size': 4,
            'max_iters': 1000,
        },
        'small': {
            'vocab_size': 5000,
            'max_seq_len': 256,
            'd_model': 384,
            'n_layers': 6,
            'n_heads': 6,
            'batch_size': 8,
            'max_iters': 10000,
        },
        'medium': {
            'vocab_size': 50257,
            'max_seq_len': 1024,
            'd_model': 768,
            'n_layers': 12,
            'n_heads': 12,
            'batch_size': 12,
            'max_iters': 100000,
        },
        'large': {
            'vocab_size': 50257,
            'max_seq_len': 2048,
            'd_model': 1024,
            'n_layers': 24,
            'n_heads': 16,
            'batch_size': 16,
            'max_iters': 500000,
        },
    }

    if model_size not in configs:
        raise ValueError(f"Invalid model_size: {model_size}. Choose from {list(configs.keys())}")

    params = configs[model_size]

    model_config = ModelConfig(
        vocab_size=params['vocab_size'],
        max_seq_len=params['max_seq_len'],
        d_model=params['d_model'],
        n_layers=params['n_layers'],
        n_heads=params['n_heads'],
    )

    training_config = TrainingConfig(
        batch_size=params['batch_size'],
        max_iters=params['max_iters'],
        device=device,
        out_dir=f'checkpoints/{model_size}_model',
    )

    data_config = DataConfig()

    return Config(
        model=model_config,
        training=training_config,
        data=data_config
    )


# Example usage and testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Load config from file
        config = load_config(sys.argv[1])
        print(f"Loaded config from {sys.argv[1]}")
        print(f"Model: {config.model.d_model}D, {config.model.n_layers} layers")
        print(f"Training: {config.training.max_iters} iterations on {config.training.device}")
    else:
        # Create default config
        config = create_default_config('tiny', 'cpu')
        print("Created default tiny config")
        print(f"Model: {config.model.d_model}D, {config.model.n_layers} layers")
        print(f"d_ff: {config.model.d_ff} (φ-harmonic scaling)")
