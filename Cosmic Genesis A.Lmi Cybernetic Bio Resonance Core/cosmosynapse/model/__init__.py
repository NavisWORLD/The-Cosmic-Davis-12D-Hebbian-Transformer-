"""
Cosmo's — The Unified 54D CST Transformer
==========================================

Fuses the 12D Cosmic Synapse Transformer and 42D Hyper-Cosmic Transformer
into a single, superior architecture with:
- 54D per-token internal state (12 scalar + 42 vector)
- Mixture-of-States Hebbian Attention
- SwiGLU feed-forward with φ-scaling
- 7-fold Lyapunov-stabilized Lorenz chaos
- Persistent episodic memory with dream consolidation
- RoPE for infinite context
- cosmos Plugin integration

Author: Cory Shane Davis
Theory: 12D Cosmic Synapse Theory (2018-2025)
Model:  Cosmo's Unified Architecture v2.0
"""

from .cosmos_config import CosmosConfig
from .cosmos_model import (
    CosmosTransformer,
    CosmosLayer,
    MixtureOfStatesAttention,
    CosmosInternalDynamics,
    PhiGatedFFN,
    PersistentEpisodicMemory,
    ChaosEnsemble,
    RMSNorm,
)

__all__ = [
    "CosmosConfig",
    "CosmosTransformer",
    "CosmosLayer",
    "MixtureOfStatesAttention",
    "CosmosInternalDynamics",
    "PhiGatedFFN",
    "PersistentEpisodicMemory",
    "ChaosEnsemble",
    "RMSNorm",
]
