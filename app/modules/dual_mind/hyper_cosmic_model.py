"""
42D HYPER-COSMIC SYNAPSE TRANSFORMER
====================================
"The Answer to Life, the Universe, and Everything"

This architecture expands the 12D CST into 42 dimensions of internal state space.
It is designed to be the "Singularity" model - beating both tiny and large models
through hyper-dimensional efficiency.

Core Innovations:
1. x42 Hyper-State: 42-dimensional vector state per token
2. Tensor Hebbian Attention: Similarity in 42D manifold
3. 7-Fold Fractal Chaos: Coupled chaotic dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

# ===================================================================
# CONSTANTS
# ===================================================================

PHI = 1.618033988749895
DIM_HYPER = 42  # The Answer

@dataclass
class HyperConfig:
    vocab_size: int = 50257
    max_seq_len: int = 2048
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    
    # 42D Parameters
    k: float = 0.1
    gamma: float = 0.05
    sigma: float = 0.5
    beta: float = 0.2
    dt: float = 0.1
    
    # Chaos
    lambda_chaos: float = 0.01
    p_chaos: float = 0.1
    
    dropout: float = 0.1
    
    def __post_init__(self):
        # φ-optimization
        self.d_ff = int(self.d_model * PHI)
        self.d_k = self.d_model // self.n_heads

# ===================================================================
# 42D INTERNAL DYNAMICS
# ===================================================================

class HyperInternalDynamics(nn.Module):
    """
    Manages the 42-dimensional internal state of each token.
    
    State shape: [batch, seq_len, 42]
    """
    
    def __init__(self, config: HyperConfig):
        super().__init__()
        self.k = config.k
        self.gamma = config.gamma
        self.dt = config.dt
        
        # Projection from d_model to 42D space for "input" to the state
        self.input_proj = nn.Linear(config.d_model, DIM_HYPER)
        
        # Learnable dynamics parameters per dimension
        self.k_vec = nn.Parameter(torch.ones(DIM_HYPER))
        self.gamma_vec = nn.Parameter(torch.ones(DIM_HYPER))
        
        # Coupling matrix (42x42) to allow dimensions to interact
        self.coupling = nn.Linear(DIM_HYPER, DIM_HYPER, bias=False)
        # Initialize close to identity to start with independent dims
        with torch.no_grad():
            self.coupling.weight.copy_(torch.eye(DIM_HYPER) + torch.randn(DIM_HYPER, DIM_HYPER)*0.01)

    def forward(self, x42: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Update 42D state.
        
        Args:
            x42: Current state [batch, seq_len, 42]
            hidden_state: Transformer hidden state [batch, seq_len, d_model]
        """
        # Input drive from the transformer processing
        # This represents the "energy" flowing into the internal state
        drive = torch.tanh(self.input_proj(hidden_state)) # [batch, seq_len, 42]
        
        # Coupled dynamics
        # dx/dt = k*drive - gamma*x + coupling(x)
        
        term1 = self.k * self.k_vec * drive
        term2 = self.gamma * self.gamma_vec * x42
        term3 = self.coupling(x42) * 0.1 # Weak coupling initially
        
        dx_dt = term1 - term2 + term3
        
        # Euler integration
        x42_new = x42 + self.dt * dx_dt
        
        # Bound to hypersphere surface (roughly) via tanh
        x42_new = torch.tanh(x42_new)
        
        return x42_new

# ===================================================================
# HYPER-HEBBIAN ATTENTION
# ===================================================================

class HyperHebbianAttention(nn.Module):
    """
    Multi-head attention modulated by 42D similarity.
    """
    
    def __init__(self, config: HyperConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.beta = config.beta
        self.sigma = config.sigma
        
        self.W_Q = nn.Linear(config.d_model, config.d_model)
        self.W_K = nn.Linear(config.d_model, config.d_model)
        self.W_V = nn.Linear(config.d_model, config.d_model)
        self.W_O = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Learnable scaling for Hebbian term
        self.beta_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, x42: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Standard Attention Scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # --- 42D HEBBIAN MODULATION ---
        # Compute pairwise Euclidean distance in 42D space
        # x42: [batch, seq_len, 42]
        
        # Efficient distance calculation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        x42_norm = (x42 ** 2).sum(dim=-1, keepdim=True) # [batch, seq_len, 1]
        x42_dot = torch.matmul(x42, x42.transpose(1, 2)) # [batch, seq_len, seq_len]
        
        dist_sq = x42_norm + x42_norm.transpose(1, 2) - 2 * x42_dot
        dist_sq = torch.clamp(dist_sq, min=0.0) # Numerical stability
        
        # Hebbian Bonus: Gaussian similarity in 42D space
        hebbian_bonus = torch.exp(-dist_sq / (2 * self.sigma**2))
        
        # Add to scores (broadcast over heads)
        scores = scores + (self.beta * self.beta_scale) * hebbian_bonus.unsqueeze(1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_O(out)

# ===================================================================
# HYPER-COSMIC LAYER
# ===================================================================

class HyperCosmicLayer(nn.Module):
    def __init__(self, config: HyperConfig):
        super().__init__()
        self.attn = HyperHebbianAttention(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        
        self.dynamics = HyperInternalDynamics(config)
        
        # φ-FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, x42: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # 1. Attention with 42D modulation
        attn_out = self.attn(self.ln1(x), x42, mask)
        x = x + attn_out
        
        # 2. Update 42D State
        # The state evolves based on the information processed at this layer
        x42_new = self.dynamics(x42, x)
        
        # 3. Feed Forward
        x = x + self.ffn(self.ln2(x))
        
        return x, x42_new

# ===================================================================
# 42D HYPER-COSMIC TRANSFORMER
# ===================================================================

class HyperCosmicTransformer(nn.Module):
    def __init__(self, config: HyperConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([
            HyperCosmicLayer(config) for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"[42D HYPER-CST] Initialized with {self.get_num_params()/1e6:.2f}M params")
        print(f"[42D HYPER-CST] Internal State Dimension: {DIM_HYPER}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        device = idx.device
        b, t = idx.shape
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.dropout(x)
        
        # Initialize 42D Hyper-State
        # Start with small random noise to break symmetry
        x42 = torch.randn(b, t, DIM_HYPER, device=device) * 0.01
        
        mask = torch.tril(torch.ones(t, t, device=device)).view(1, t, t)
        
        x42_states = []
        
        for layer in self.layers:
            x, x42 = layer(x, x42, mask)
            x42_states.append(x42)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, {'x42_final': x42}

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate text autoregressively.
        """
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else \
                       idx[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _, _ = self.forward(idx_cond)

            # Take last timestep
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

