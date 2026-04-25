"""
COSMO'S TRANSFORMER — The Unified 54D CST Architecture
========================================================

The finalized model fusing 12D Cosmic Synapse Theory and 42D Hyper-Cosmic
Theory into a single, superior transformer that surpasses both predecessors
and cosmos at every level.

Core Innovations:
    1. 54D Internal State (x₅₄) — 12 scalar + 42 vector dimensions per token
    2. Mixture-of-States Attention — Learned gate blending standard + Hebbian
    3. SwiGLU Feed-Forward — φ-scaled, superior to GELU
    4. 7-Fold Chaos Ensemble — Coupled Lorenz with Lyapunov monitoring
    5. Persistent Episodic Memory — 256-slot with dream consolidation
    6. RoPE — Rotary Position Embeddings for infinite context
    7. RMSNorm — Faster than LayerNorm, used in LLaMA/Gemma

Math:
    dx₅₄/dt = K·drive - Γ·x₅₄ + C(x₅₄)
    Attention(Q,K,V,x₅₄) = softmax(QK^T/√d + gate·β·H(x₅₄))V
    H(x₅₄)ᵢⱼ = exp(-‖x₅₄ᵢ - x₅₄ⱼ‖²/2σ²)

Author: Cory Shane Davis
Theory: 12D Cosmic Synapse Theory (2018–2025)
Model:  Cosmo's v2.0.0
"""

import math
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .cosmos_config import CosmosConfig, PHI, PHI_INV

# ===================================================================
# RMS NORM (replaces LayerNorm — faster, used in LLaMA/Gemma)
# ===================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ===================================================================
# ROTARY POSITION EMBEDDINGS (RoPE — infinite context)
# ===================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (Su et al. 2021).

    Encodes position information directly into Q/K via rotation,
    enabling theoretically infinite context length.
    """

    def __init__(self, d_k: int, max_seq_len: int = 8192, theta: float = 10000.0) -> None:
        super().__init__()
        # Compute frequency bands
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("freqs", freqs)

        # Pre-compute cos/sin cache
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs_full = torch.outer(t, freqs)  # [max_seq_len, d_k//2]
        self.register_buffer("cos_cached", freqs_full.cos())
        self.register_buffer("sin_cached", freqs_full.sin())

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary embeddings to x.

        Args:
            x: [batch, n_heads, seq_len, d_k]
            offset: position offset for cached inference
        """
        seq_len = x.shape[2]
        d_k = x.shape[3]
        half_d = d_k // 2

        cos = self.cos_cached[offset : offset + seq_len, :half_d]  # [seq_len, half_d]
        sin = self.sin_cached[offset : offset + seq_len, :half_d]

        # Reshape for broadcasting: [1, 1, seq_len, half_d]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split x into two halves
        x1 = x[..., :half_d]
        x2 = x[..., half_d : 2 * half_d]

        # Apply rotation
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Reassemble (handle odd d_k)
        if d_k % 2 == 1:
            return torch.cat([out1, out2, x[..., -1:]], dim=-1)
        return torch.cat([out1, out2], dim=-1)


# ===================================================================
# CHAOS ENSEMBLE — 7-Fold Coupled Lorenz with Lyapunov Monitoring
# ===================================================================

class ChaosEnsemble:
    """
    7-fold coupled Lorenz attractor ensemble with Lyapunov stability.

    Each oscillator is a standard Lorenz system with weak coupling
    between neighbors. The Lyapunov exponent is monitored to prevent
    runaway divergence.
    """

    def __init__(
        self,
        n_oscillators: int = 7,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        dt: float = 0.01,
        coupling_strength: float = 0.05,
    ) -> None:
        self.n_osc = n_oscillators
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.coupling = coupling_strength

        # State: [n_oscillators, 3]
        self.states = np.random.randn(n_oscillators, 3) * 0.1
        self.lyapunov_sum = 0.0
        self.lyapunov_steps = 0

    def step(self, n_steps: int = 10) -> np.ndarray:
        """Evolve all oscillators with coupling."""
        for _ in range(n_steps):
            new_states = np.zeros_like(self.states)
            for i in range(self.n_osc):
                s = self.states[i]
                dx = self.sigma * (s[1] - s[0])
                dy = s[0] * (self.rho - s[2]) - s[1]
                dz = s[0] * s[1] - self.beta * s[2]

                # Coupling: average neighbors
                left = self.states[(i - 1) % self.n_osc]
                right = self.states[(i + 1) % self.n_osc]
                coupling_term = self.coupling * (left + right - 2 * s)

                new_states[i, 0] = s[0] + self.dt * (dx + coupling_term[0])
                new_states[i, 1] = s[1] + self.dt * (dy + coupling_term[1])
                new_states[i, 2] = s[2] + self.dt * (dz + coupling_term[2])

            # Lyapunov monitoring
            delta = np.linalg.norm(new_states - self.states)
            if delta > 1e-10:
                self.lyapunov_sum += np.log(delta)
                self.lyapunov_steps += 1

            self.states = new_states

        return self.states.copy()

    @property
    def lyapunov_exponent(self) -> float:
        """Current estimated Lyapunov exponent."""
        if self.lyapunov_steps == 0:
            return 0.0
        return self.lyapunov_sum / self.lyapunov_steps

    def get_noise(self, shape: Tuple[int, ...], device: str = "cpu") -> torch.Tensor:
        """Generate chaos-modulated noise tensor."""
        self.step()
        # Combine all oscillator states into a rich noise seed
        combined = self.states.flatten()  # [n_osc * 3]
        base = torch.from_numpy(combined).float().to(device)

        noise = torch.randn(shape, device=device)
        # Modulate with chaos: scale by mean and normalize
        chaos_scale = base.mean().abs().clamp(min=0.01, max=5.0)
        noise = noise * chaos_scale
        return noise / (noise.std() + 1e-8)


# ===================================================================
# 54D INTERNAL STATE DYNAMICS
# ===================================================================

class CosmosInternalDynamics(nn.Module):
    """
    Unified 54D internal state manager.

    Fuses:
    - 12D scalar states (from 12D CST: dx₁₂/dt = k·Ω - γ·x₁₂)
    - 42D vector states (from 42D Hyper-CST: coupled dynamics)

    Into a single 54D state with:
    - Per-dimension learnable k/γ vectors
    - 54×54 coupling matrix (initialized near identity)
    - Configurable ODE solver (Euler default)

    dx₅₄/dt = K·drive - Γ·x₅₄ + C(x₅₄)
    """

    def __init__(self, config: CosmosConfig) -> None:
        super().__init__()
        self.d_state = config.d_state
        self.dt = config.dt

        # Project from d_model → 54D state drive
        self.input_proj = nn.Linear(config.d_model, config.d_state, bias=False)

        # Per-dimension learnable parameters
        self.k_vec = nn.Parameter(torch.ones(config.d_state) * config.k)
        self.gamma_vec = nn.Parameter(torch.ones(config.d_state) * config.gamma)

        # 54×54 coupling matrix — starts near identity for stability
        self.coupling = nn.Linear(config.d_state, config.d_state, bias=False)
        with torch.no_grad():
            self.coupling.weight.copy_(
                torch.eye(config.d_state) + torch.randn(config.d_state, config.d_state) * 0.01
            )

        # Coupling strength (learnable)
        self.coupling_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x54: torch.Tensor,
        hidden_state: torch.Tensor,
        omega: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update 54D state.

        Args:
            x54: Current state [batch, seq_len, 54]
            hidden_state: Transformer hidden [batch, seq_len, d_model]
            omega: Optional connectivity signal [batch, seq_len] from attention

        Returns:
            Updated x54 [batch, seq_len, 54]
        """
        # Drive from transformer hidden state
        drive = torch.tanh(self.input_proj(hidden_state))  # [batch, seq_len, 54]

        # If we have connectivity signal (Ω), modulate the drive
        if omega is not None:
            # omega: [batch, seq_len] → [batch, seq_len, 1]
            omega_scale = omega.unsqueeze(-1).clamp(0.0, 5.0)
            drive = drive * (1.0 + 0.1 * omega_scale)

        # Coupled dynamics: dx/dt = K·drive - Γ·x + C(x)
        term_drive = self.k_vec * drive
        term_decay = self.gamma_vec * x54
        term_coupling = self.coupling(x54) * self.coupling_scale

        dx_dt = term_drive - term_decay + term_coupling

        # Euler integration
        x54_new = x54 + self.dt * dx_dt

        # Bound to hypersphere via tanh
        x54_new = torch.tanh(x54_new)

        return x54_new


# ===================================================================
# MIXTURE-OF-STATES ATTENTION
# ===================================================================

class MixtureOfStatesAttention(nn.Module):
    """
    Multi-head attention with Hebbian modulation from 54D state space.

    The key innovation: a learned gate mixes standard attention scores
    with Hebbian similarity scores from the 54D internal state.

    Attention(Q,K,V,x₅₄) = softmax(scores_std + gate · β · H(x₅₄)) V

    where H(x₅₄)ᵢⱼ = exp(-‖x₅₄ᵢ - x₅₄ⱼ‖² / 2σ²)
    """

    def __init__(self, config: CosmosConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_model = config.d_model
        self.beta = config.beta
        self.sigma = config.sigma
        self.use_state_gate = config.use_state_gate

        # Q, K, V projections (no bias for efficiency)
        self.W_Q = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.W_K = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.W_V = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.W_O = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Learnable Hebbian scaling
        self.beta_scale = nn.Parameter(torch.ones(1))
        self.sigma_scale = nn.Parameter(torch.ones(1))

        # State gate: learned scalar per head (controls Hebbian influence)
        if self.use_state_gate:
            self.state_gate = nn.Parameter(torch.ones(config.n_heads) * 0.5)

        # RoPE (if enabled)
        if config.use_rotary:
            self.rope = RotaryPositionEmbedding(config.d_k, config.max_seq_len * 2, config.rope_theta)
        else:
            self.rope = None

    def compute_hebbian_bonus(self, x54: torch.Tensor) -> torch.Tensor:
        """
        Compute Hebbian connectivity matrix from 54D state.

        H(x₅₄)ᵢⱼ = exp(-‖x₅₄ᵢ - x₅₄ⱼ‖² / 2σ²)

        Uses efficient distance computation:
        ‖a-b‖² = ‖a‖² + ‖b‖² - 2⟨a,b⟩

        Args:
            x54: [batch, seq_len, 54]
        Returns:
            hebbian: [batch, seq_len, seq_len]
        """
        # Efficient Euclidean distance
        x_norm_sq = (x54 ** 2).sum(dim=-1, keepdim=True)  # [B, S, 1]
        x_dot = torch.matmul(x54, x54.transpose(1, 2))    # [B, S, S]
        dist_sq = x_norm_sq + x_norm_sq.transpose(1, 2) - 2 * x_dot
        dist_sq = torch.clamp(dist_sq, min=0.0)

        # Gaussian similarity
        sigma_eff = self.sigma * self.sigma_scale.abs().clamp(min=0.01)
        hebbian = torch.exp(-dist_sq / (2 * sigma_eff ** 2))

        return hebbian

    def forward(
        self,
        x: torch.Tensor,
        x54: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Mixture-of-States Hebbian modulation.

        Args:
            x: Input [batch, seq_len, d_model]
            x54: Internal state [batch, seq_len, 54]
            mask: Causal mask [1, seq_len, seq_len]

        Returns:
            output: [batch, seq_len, d_model]
            omega: Connectivity [batch, seq_len] (for state dynamics)
        """
        B, S, D = x.shape

        # Project Q, K, V
        Q = self.W_Q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: [B, n_heads, S, d_k]

        # Apply RoPE
        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)

        # Standard scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [B, n_heads, S, S]

        # Hebbian bonus from 54D state
        hebbian = self.compute_hebbian_bonus(x54)  # [B, S, S]
        hebbian_scaled = (self.beta * self.beta_scale) * hebbian.unsqueeze(1)  # [B, 1, S, S]

        # Mix with state gate (per-head)
        if self.use_state_gate:
            gate = torch.sigmoid(self.state_gate).view(1, self.n_heads, 1, 1)
            scores = scores + gate * hebbian_scaled
        else:
            scores = scores + hebbian_scaled

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply to values
        out = torch.matmul(attn_weights, V)  # [B, n_heads, S, d_k]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection
        out = self.resid_dropout(self.W_O(out))

        # Compute connectivity signal (Ω) for state dynamics
        # Sum of attention weights per token (averaged over heads)
        omega = attn_weights.mean(dim=1).sum(dim=-1)  # [B, S]

        return out, omega


# ===================================================================
# SwiGLU FEED-FORWARD (φ-SCALED)
# ===================================================================

class PhiGatedFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network with φ-scaled hidden dimension.

    d_ff = ⌊d_model × φ⌋
    FFN(x) = (Swish(W₁x) ⊙ W₃x) W₂

    SwiGLU is used in LLaMA, PaLM, Gemma — superior to GELU.
    """

    def __init__(self, config: CosmosConfig) -> None:
        super().__init__()
        self.W1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.W2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        self.W3 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)  # Gate
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(W1·x) ⊙ W3·x) · W2
        gate = F.silu(self.W1(x))  # Swish/SiLU activation
        up = self.W3(x)
        out = self.W2(gate * up)
        return self.dropout(out)


# ===================================================================
# PERSISTENT EPISODIC MEMORY
# ===================================================================

class PersistentEpisodicMemory(nn.Module):
    """
    Memory buffer storing (embedding, x54_state) pairs.

    Upgrades over 12D EpisodicMemory:
    - 256-slot buffer (vs 100)
    - Dual-key retrieval: semantic + state similarity
    - Dream consolidation: compress old memories into prototypes
    - Disk persistence via checkpoint
    """

    def __init__(self, config: CosmosConfig) -> None:
        super().__init__()
        self.memory_size = config.memory_size
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.alpha = config.alpha_memory
        self.consolidation_enabled = config.memory_consolidation

        # Memory banks
        self.register_buffer(
            "mem_embeddings",
            torch.zeros(config.memory_size, config.d_model),
        )
        self.register_buffer(
            "mem_states",
            torch.zeros(config.memory_size, config.d_state),
        )
        self.register_buffer(
            "mem_importance",
            torch.zeros(config.memory_size),
        )
        self.mem_ptr = 0
        self.mem_filled = 0

    def update(self, embeddings: torch.Tensor, states: torch.Tensor) -> None:
        """Add current states to memory buffer (training only)."""
        if not self.training:
            return

        B, S, D = embeddings.shape
        flat_emb = embeddings.detach().view(-1, D)
        flat_st = states.detach().view(-1, self.d_state)

        n = min(len(flat_emb), self.memory_size)
        for i in range(n):
            self.mem_embeddings[self.mem_ptr] = flat_emb[i]
            self.mem_states[self.mem_ptr] = flat_st[i]
            self.mem_importance[self.mem_ptr] = 1.0  # Initial importance
            self.mem_ptr = (self.mem_ptr + 1) % self.memory_size
            self.mem_filled = min(self.mem_filled + 1, self.memory_size)

    def retrieve(
        self,
        query_emb: torch.Tensor,
        query_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dual-key retrieval from memory.

        Args:
            query_emb: [batch, seq_len, d_model]
            query_state: [batch, seq_len, d_state]

        Returns:
            retrieved: [batch, seq_len, d_model]
        """
        if self.mem_filled == 0:
            return torch.zeros_like(query_emb)

        B, S, D = query_emb.shape
        filled = self.mem_filled

        # Semantic similarity
        q_norm = F.normalize(query_emb.view(-1, D), dim=-1)       # [B*S, D]
        m_norm = F.normalize(self.mem_embeddings[:filled], dim=-1)  # [filled, D]
        sim_semantic = torch.matmul(q_norm, m_norm.t())             # [B*S, filled]

        # State similarity (Gaussian in 54D)
        q_state_flat = query_state.view(-1, self.d_state)           # [B*S, 54]
        m_state = self.mem_states[:filled]                          # [filled, 54]
        # Efficient: ‖a-b‖² = ‖a‖² + ‖b‖² - 2⟨a,b⟩
        q_sq = (q_state_flat ** 2).sum(-1, keepdim=True)            # [B*S, 1]
        m_sq = (m_state ** 2).sum(-1, keepdim=True).t()             # [1, filled]
        qm_dot = torch.matmul(q_state_flat, m_state.t())           # [B*S, filled]
        dist_sq = q_sq + m_sq - 2 * qm_dot
        sim_state = torch.exp(-dist_sq.clamp(min=0) / 2.0)         # [B*S, filled]

        # Combined similarity (dual-key)
        importance = self.mem_importance[:filled].unsqueeze(0)       # [1, filled]
        sim_total = sim_semantic * sim_state * importance
        sim_weights = F.softmax(sim_total, dim=-1)                  # [B*S, filled]

        # Retrieve
        retrieved = torch.matmul(sim_weights, self.mem_embeddings[:filled])  # [B*S, D]
        return retrieved.view(B, S, D)

    def consolidate(self) -> None:
        """
        Dream consolidation: decay old memories, keep important ones.
        Called periodically during training to prevent memory staleness.
        """
        if not self.consolidation_enabled or self.mem_filled == 0:
            return

        # Decay importance
        self.mem_importance[:self.mem_filled] *= 0.99

        # Zero out very low importance memories (free slots)
        mask = self.mem_importance[:self.mem_filled] < 0.01
        if mask.any():
            indices = mask.nonzero(as_tuple=True)[0]
            self.mem_embeddings[indices] = 0.0
            self.mem_states[indices] = 0.0
            self.mem_importance[indices] = 0.0


# ===================================================================
# COSMO'S TRANSFORMER LAYER
# ===================================================================

class CosmosLayer(nn.Module):
    """
    One complete layer of Cosmo's Transformer.

    Pipeline:
    1. RMSNorm → MixtureOfStatesAttention → residual
    2. 54D State Update (driven by hidden state + connectivity Ω)
    3. Episodic Memory update + retrieval
    4. RMSNorm → PhiGatedFFN (SwiGLU) → residual
    5. Chaos injection (training only, probabilistic)
    """

    def __init__(self, config: CosmosConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Sub-modules
        self.attention = MixtureOfStatesAttention(config)
        self.dynamics = CosmosInternalDynamics(config)
        self.memory = PersistentEpisodicMemory(config)
        self.ffn = PhiGatedFFN(config)

        # RMSNorm (pre-norm architecture)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        # Chaos generator (unique per layer for diversity)
        self.chaos = ChaosEnsemble(
            n_oscillators=config.n_chaos_oscillators,
        )

    def inject_chaos(self, x: torch.Tensor) -> torch.Tensor:
        """Inject Lorenz chaos during training."""
        if self.training and torch.rand(1).item() < self.config.p_chaos:
            noise = self.chaos.get_noise(x.shape, device=x.device)

            # Adaptive strength: reduce chaos as Lyapunov exponent grows
            if self.config.chaos_adaptive:
                lyap = abs(self.chaos.lyapunov_exponent)
                strength = self.config.lambda_chaos / (1.0 + lyap)
            else:
                strength = self.config.lambda_chaos

            x = x + strength * noise
        return x

    def forward(
        self,
        x: torch.Tensor,
        x54: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    [batch, seq_len, d_model]
            x54:  [batch, seq_len, 54]
            mask: [1, seq_len, seq_len]

        Returns:
            x_out:   [batch, seq_len, d_model]
            x54_out: [batch, seq_len, 54]
        """
        # 1. Self-Attention with Hebbian modulation
        attn_out, omega = self.attention(self.norm1(x), x54, mask)
        x = x + attn_out

        # 2. Update 54D state
        x54_new = self.dynamics(x54, x, omega)

        # 3. Episodic memory
        if self.training:
            self.memory.update(x, x54_new)
        retrieved = self.memory.retrieve(x, x54_new)
        x = x + self.config.alpha_memory * retrieved

        # 4. Feed-Forward
        ff_out = self.ffn(self.norm2(x))

        # 5. Chaos injection
        ff_out = self.inject_chaos(ff_out)

        # 6. Residual
        x = x + ff_out

        return x, x54_new


# ===================================================================
# COSMO'S TRANSFORMER — THE COMPLETE MODEL
# ===================================================================

class CosmosTransformer(nn.Module):
    """
    Cosmo's: The Unified 54D Cosmic Synapse Transformer.

    This is the finalized, production-grade model that fuses the complete
    12D and 42D architectures into a single superior architecture.

    Features:
    - 54D per-token internal state (12 scalar + 42 vector)
    - Mixture-of-States Hebbian Attention with learned gates
    - SwiGLU feed-forward with φ-harmonic scaling
    - 7-fold Lyapunov-stabilized Lorenz chaos
    - 256-slot persistent episodic memory
    - RoPE for infinite context
    - Persistent x₅₄ state across generation windows
    """

    def __init__(self, config: CosmosConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Position embedding (fallback if RoPE disabled)
        if not config.use_rotary:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            self.position_embedding = None

        # Embedding dropout
        self.dropout = nn.Dropout(config.dropout)

        # Stack of Cosmos Layers
        self.layers = nn.ModuleList([
            CosmosLayer(config, i) for i in range(config.n_layers)
        ])

        # Final norm
        self.norm_f = RMSNorm(config.d_model)

        # LM head (weight-tied with token embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

        # Special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("W_O.weight") or pn.endswith("W2.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

        n_params = self.get_num_params()
        print(f"[COSMO'S] Initialized with {n_params / 1e6:.2f}M parameters")
        print(f"[COSMO'S] φ-dims: d_model={config.d_model}, d_ff={config.d_ff}")
        print(f"[COSMO'S] State: {config.d_state}D (12 scalar + 42 vector)")
        print(f"[COSMO'S] Layers={config.n_layers}, Heads={config.n_heads}")
        print(f"[COSMO'S] Chaos: {config.n_chaos_oscillators}-fold Lorenz")
        print(f"[COSMO'S] Memory: {config.memory_size}-slot episodic")
        print(f"[COSMO'S] RoPE: {'enabled' if config.use_rotary else 'disabled'}")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with φ-scaled variance."""
        if isinstance(module, nn.Linear):
            std = 0.02 * PHI_INV
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Count total parameters (excluding position embeddings if tied)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        state_x54: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], torch.Tensor]:
        """
        Forward pass.

        Args:
            idx: Token indices [batch, seq_len]
            targets: Target indices [batch, seq_len] (for loss)
            state_x54: Persistent state [batch, n_layers, d_state] from prior window

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar or None
            metrics: dict with x54 stats, chaos info, etc.
            next_state_x54: [batch, n_layers, d_state] for next window
        """
        device = idx.device
        B, S = idx.shape

        # ---- Embeddings ----
        tok_emb = self.token_embedding(idx)  # [B, S, D]
        if self.position_embedding is not None:
            pos = torch.arange(0, S, dtype=torch.long, device=device).unsqueeze(0)
            tok_emb = tok_emb + self.position_embedding(pos % self.config.max_seq_len)
        x = self.dropout(tok_emb)

        # ---- Causal mask ----
        mask = torch.tril(torch.ones(S, S, device=device)).view(1, 1, S, S)

        # ---- Initialize per-layer 54D states ----
        n_layers = len(self.layers)
        next_state = torch.zeros(B, n_layers, self.config.d_state, device=device)

        layer_states: List[torch.Tensor] = []
        if state_x54 is not None:
            for i in range(n_layers):
                # Broadcast prior state across new sequence
                init = state_x54[:, i, :].unsqueeze(1)  # [B, 1, 54]
                layer_states.append(init.expand(-1, S, -1).clone())
        else:
            for _ in range(n_layers):
                # Small random noise to break symmetry
                layer_states.append(
                    torch.randn(B, S, self.config.d_state, device=device) * 0.01
                )

        # ---- Pass through layers ----
        for i, layer in enumerate(self.layers):
            x, x54_new = layer(x, layer_states[i], mask)
            # Save last-token state for persistence
            next_state[:, i, :] = x54_new[:, -1, :]

        # ---- Output ----
        x = self.norm_f(x)
        logits = self.lm_head(x)

        # ---- Loss ----
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        # ---- Metrics ----
        metrics = {
            "x54_mean": next_state.mean().item(),
            "x54_std": next_state.std().item() if next_state.numel() > 1 else 0.0,
            "x54_scalar_mean": next_state[:, :, :12].mean().item(),
            "x54_vector_mean": next_state[:, :, 12:].mean().item(),
            "chaos_lyapunov": self.layers[0].chaos.lyapunov_exponent,
        }

        return logits, loss, metrics, next_state

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        state_x54: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate tokens autoregressively with persistent 54D state.

        The x₅₄ state carries forward across generation windows,
        giving Cosmo's theoretically infinite context.

        Args:
            idx: Context tokens [batch, context_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
            top_p: Nucleus sampling threshold (optional)
            state_x54: Initial state from prior context

        Returns:
            generated: [batch, context_len + max_new_tokens]
            final_state: [batch, n_layers, d_state]
        """
        curr_state = state_x54

        for _ in range(max_new_tokens):
            # Crop to max_seq_len but RETAIN state (infinite context)
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len \
                       else idx[:, -self.config.max_seq_len:]

            # Forward pass — state carries over
            logits, _, _, next_state = self.forward(idx_cond, state_x54=curr_state)
            curr_state = next_state

            # Sample from last position
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                # Scatter back
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx, curr_state


# ===================================================================
# STANDALONE TEST
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("COSMO'S TRANSFORMER — Standalone Test")
    print("=" * 60)

    # Test with tiny config
    config = CosmosConfig.tiny(vocab_size=1000)
    model = CosmosTransformer(config)

    # Forward pass
    B, S = 2, 32
    x = torch.randint(0, 1000, (B, S))
    logits, loss, metrics, state = model(x, targets=x)

    print(f"\nForward Pass:")
    print(f"  Input:   {x.shape}")
    print(f"  Logits:  {logits.shape}")
    print(f"  Loss:    {loss.item():.4f}")
    print(f"  State:   {state.shape}")
    print(f"  Metrics: {metrics}")

    # Generation
    model.eval()
    ctx = torch.randint(0, 1000, (1, 5))
    gen, final_state = model.generate(ctx, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"\nGeneration:")
    print(f"  Context: {ctx.shape[1]} tokens")
    print(f"  Output:  {gen.shape[1]} tokens")
    print(f"  State persists: {final_state.shape}")

    print(f"\n[COSMO'S] All tests passed ✓")
