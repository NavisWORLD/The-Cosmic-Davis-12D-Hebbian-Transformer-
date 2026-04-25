"""
COSMO'S CONFIGURATION
=====================
Unified configuration for the 54D Cosmic Synapse Transformer.

Presets:
    CosmosConfig.tiny()   — ~4M params   (testing/prototyping)
    CosmosConfig.small()  — ~30M params  (local training)
    CosmosConfig.medium() — ~125M params (production)
    CosmosConfig.large()  — ~350M params (maximum quality)

Author: Cory Shane Davis
"""

import math
from dataclasses import dataclass, field
from typing import Optional

# ===================================================================
# CONSTANTS
# ===================================================================

PHI = 1.618033988749895          # Golden Ratio
PHI_INV = 1.0 / PHI             # Inverse Golden Ratio
DIM_SCALAR = 12                  # 12D scalar state dimensions
DIM_VECTOR = 42                  # 42D vector state dimensions
DIM_STATE = DIM_SCALAR + DIM_VECTOR  # 54D total internal state


@dataclass
class CosmosConfig:
    """
    Unified configuration for Cosmo's Transformer.

    Merges the best of CosmicConfig (12D) and HyperConfig (42D)
    with significant upgrades.
    """

    # ---- Model Architecture ----
    vocab_size: int = 50257          # Default GPT-2 tokenizer size
    max_seq_len: int = 2048          # Maximum sequence length
    d_model: int = 768               # Will be φ-optimized
    n_layers: int = 12               # Number of transformer layers
    n_heads: int = 12                # Number of attention heads

    # ---- 54D Internal State Parameters ----
    d_state: int = DIM_STATE         # 54D total (12 scalar + 42 vector)
    d_state_scalar: int = DIM_SCALAR # 12D scalar subspace
    d_state_vector: int = DIM_VECTOR # 42D vector subspace
    k: float = 0.1                   # State coupling constant
    gamma: float = 0.05              # State decay rate
    dt: float = 0.1                  # State update timestep

    # ---- Hebbian Attention Parameters ----
    sigma: float = 0.5               # Hebbian similarity spread
    beta: float = 0.2                # Hebbian attention weight
    use_state_gate: bool = True      # Learned gate for Hebbian mixing

    # ---- Chaos Parameters ----
    n_chaos_oscillators: int = 7     # 7-fold Lorenz coupling
    lambda_chaos: float = 0.01       # Chaos injection strength
    p_chaos: float = 0.1             # Probability of chaos injection
    chaos_adaptive: bool = True      # Adapt chaos strength to loss

    # ---- Memory Parameters ----
    memory_size: int = 256           # Episodic memory buffer size
    alpha_memory: float = 0.1        # Memory adaptation rate
    memory_consolidation: bool = True  # Enable dream-style consolidation

    # ---- Positional Encoding ----
    use_rotary: bool = True          # RoPE for infinite context
    rope_theta: float = 10000.0      # RoPE frequency base

    # ---- Regularisation ----
    dropout: float = 0.1
    use_bias: bool = False           # No bias (modern practice)

    # ---- Compute ----
    use_flash_attn: bool = False     # Flash Attention (if available)

    # ---- Derived (computed in __post_init__) ----
    d_ff: int = 0
    d_k: int = 0

    def __post_init__(self) -> None:
        # φ-optimize d_model
        self.d_model = self._phi_optimize(self.d_model)
        # Ensure divisible by n_heads
        self.d_model = (self.d_model // self.n_heads) * self.n_heads
        # φ-scaled feed-forward dimension
        self.d_ff = int(self.d_model * PHI)
        # Per-head dimension
        self.d_k = self.d_model // self.n_heads

    @staticmethod
    def _phi_optimize(d: int) -> int:
        """Round dimension to nearest φ-harmonic value."""
        n = round(math.log(max(d, 1)) / math.log(PHI))
        return max(int(PHI ** n), 8)  # Floor at 8

    # ---- Preset Configurations ----

    @classmethod
    def tiny(cls, vocab_size: int = 50257) -> "CosmosConfig":
        """~4M parameter config for testing."""
        return cls(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=4,
            n_heads=4,
            memory_size=64,
            n_chaos_oscillators=3,
            max_seq_len=512,
        )

    @classmethod
    def small(cls, vocab_size: int = 50257) -> "CosmosConfig":
        """~30M parameter config for local training."""
        return cls(
            vocab_size=vocab_size,
            d_model=384,
            n_layers=8,
            n_heads=8,
            memory_size=128,
            n_chaos_oscillators=5,
            max_seq_len=1024,
        )

    @classmethod
    def medium(cls, vocab_size: int = 50257) -> "CosmosConfig":
        """~125M parameter config for production."""
        return cls(
            vocab_size=vocab_size,
            d_model=768,
            n_layers=12,
            n_heads=12,
            memory_size=256,
            n_chaos_oscillators=7,
            max_seq_len=2048,
        )

    @classmethod
    def large(cls, vocab_size: int = 50257) -> "CosmosConfig":
        """~350M parameter config for maximum quality."""
        return cls(
            vocab_size=vocab_size,
            d_model=1024,
            n_layers=24,
            n_heads=16,
            memory_size=512,
            n_chaos_oscillators=7,
            max_seq_len=4096,
        )
