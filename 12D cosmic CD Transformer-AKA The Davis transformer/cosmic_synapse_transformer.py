"""
12D COSMIC SYNAPSE TRANSFORMER
================================
A production-grade transformer architecture implementing the complete 
12-Dimensional Cosmic Synapse Theory by Cory Shane Davis.

This is NOT a simulation - this is a real, trainable model that implements:
- φ-harmonic dimensional scaling
- Per-token adaptive internal states (x₁₂)
- Hebbian-modulated attention
- Chaos-guided exploration
- Memory-augmented processing
- Golden ratio optimization throughout

Author: Cory Shane Davis
Theory: 12D Cosmic Synapse Theory (2018-2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

# ===================================================================
# CONSTANTS & CONFIGURATION
# ===================================================================

PHI = 1.618033988749895  # Golden Ratio
PHI_INV = 1 / PHI

@dataclass
class CosmicConfig:
    """Configuration for 12D Cosmic Synapse Transformer"""
    
    # Model architecture
    vocab_size: int = 50257  # GPT-2 vocab size
    max_seq_len: int = 2048
    d_model: int = 768  # Will be φ-optimized
    n_layers: int = 12
    n_heads: int = 12
    
    # 12D CST Parameters
    k: float = 0.1  # Internal state coupling constant
    gamma: float = 0.05  # Internal state decay
    sigma: float = 0.5  # Hebbian similarity spread
    beta: float = 0.2  # Hebbian attention weight
    dt: float = 0.1  # Internal state update timestep
    
    # Chaos parameters
    lambda_chaos: float = 0.01  # Chaos injection strength
    p_chaos: float = 0.1  # Probability of chaos injection during training
    
    # Training
    dropout: float = 0.1
    use_bias: bool = True
    
    # Memory
    memory_size: int = 100  # Size of episodic memory buffer
    alpha_memory: float = 0.1  # Memory adaptation rate
    
    def __post_init__(self) -> None:
        # Ensure d_model is φ-optimized
        self.d_model = self._phi_optimize(self.d_model)
        # Compute feed-forward dimension
        self.d_ff = int(self.d_model * PHI)
        # Ensure divisibility by number of heads
        self.d_model = (self.d_model // self.n_heads) * self.n_heads
        self.d_k = self.d_model // self.n_heads
    
    @staticmethod
    def _phi_optimize(d: int) -> int:
        """Round dimension to nearest φ-harmonic value"""
        # Find closest φ^n
        n = round(math.log(d) / math.log(PHI))
        return int(PHI ** n)

# ===================================================================
# LORENZ CHAOS GENERATOR
# ===================================================================

class LorenzAttractor:
    """Generates chaotic sequences for exploration"""
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0, dt: float = 0.01) -> None:
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.state = np.random.randn(3)

    def step(self, n_steps: int = 10) -> np.ndarray:
        """Evolve the Lorenz attractor"""
        for _ in range(n_steps):
            dx = self.sigma * (self.state[1] - self.state[0])
            dy = self.state[0] * (self.rho - self.state[2]) - self.state[1]
            dz = self.state[0] * self.state[1] - self.beta * self.state[2]

            self.state[0] += self.dt * dx
            self.state[1] += self.dt * dy
            self.state[2] += self.dt * dz

        return self.state.copy()

    def get_noise(self, shape: Tuple[int, ...], device: str = 'cpu') -> torch.Tensor:
        """Generate chaotic noise tensor"""
        self.step()
        # Project 3D Lorenz to desired shape
        base = torch.from_numpy(self.state).float().to(device)
        noise = torch.randn(shape, device=device)
        # Modulate by Lorenz state
        noise = noise * base.mean()
        return noise / (noise.std() + 1e-8)

# ===================================================================
# INTERNAL STATE DYNAMICS (12th Dimension)
# ===================================================================

class InternalStateDynamics(nn.Module):
    """
    Implements the 12th dimension: adaptive internal state per token.
    
    dx₁₂/dt = k·Ω - γ·x₁₂
    
    This gives each token a "cognitive state" that evolves based on
    its connectivity to other tokens (Hebbian principle).
    """
    
    def __init__(self, config: CosmicConfig) -> None:
        super().__init__()
        self.k = config.k
        self.gamma = config.gamma
        self.dt = config.dt

        # Learnable scaling factors
        self.k_scale = nn.Parameter(torch.ones(1))
        self.gamma_scale = nn.Parameter(torch.ones(1))

    def forward(self, x12_current: torch.Tensor, omega_connectivity: torch.Tensor) -> torch.Tensor:
        """
        Update internal states based on connectivity.

        Args:
            x12_current: Current internal states [batch, seq_len]
            omega_connectivity: Total connectivity per token [batch, seq_len]

        Returns:
            Updated x12 states
        """
        # Compute rate of change
        dx12_dt = (self.k * self.k_scale) * omega_connectivity - \
                  (self.gamma * self.gamma_scale) * x12_current

        # Euler integration
        x12_new = x12_current + self.dt * dx12_dt

        # Bound to [-1, 1] via tanh
        x12_new = torch.tanh(x12_new)

        return x12_new

# ===================================================================
# HEBBIAN ATTENTION MECHANISM
# ===================================================================

class HebbianMultiHeadAttention(nn.Module):
    """
    Multi-head attention with Hebbian connectivity modulation.
    
    Standard attention + Hebbian bonus based on x₁₂ similarity:
    Attention(Q,K,V,x₁₂) = softmax(QK^T/√d_k + β·H(x₁₂))V
    
    where H(x₁₂)ᵢⱼ = exp(-(x₁₂ᵢ - x₁₂ⱼ)²/2σ²)
    """
    
    def __init__(self, config: CosmicConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_model = config.d_model
        self.beta = config.beta
        self.sigma = config.sigma

        # Q, K, V projections
        self.W_Q = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.W_K = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.W_V = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)

        # Output projection
        self.W_O = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Learnable Hebbian weight
        self.beta_scale = nn.Parameter(torch.ones(1))
        self.sigma_scale = nn.Parameter(torch.ones(1))

    def compute_hebbian_bonus(self, x12: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Compute Hebbian connectivity bonus matrix.

        H(x₁₂)ᵢⱼ = exp(-(x₁₂ᵢ - x₁₂ⱼ)²/2σ²)
        """
        # x12: [batch, seq_len]
        x12_i = x12.unsqueeze(2)  # [batch, seq_len, 1]
        x12_j = x12.unsqueeze(1)  # [batch, 1, seq_len]

        # Compute squared difference
        x12_diff_sq = (x12_i - x12_j) ** 2

        # Gaussian similarity
        sigma_eff = self.sigma * self.sigma_scale
        hebbian_bonus = torch.exp(-x12_diff_sq / (2 * sigma_eff ** 2))

        return hebbian_bonus

    def forward(self, x: torch.Tensor, x12: torch.Tensor, mask: Optional[torch.Tensor] = None, return_omega: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with Hebbian modulation.
        
        Args:
            x: Input embeddings [batch, seq_len, d_model]
            x12: Internal states [batch, seq_len]
            mask: Attention mask [batch, seq_len, seq_len]
            return_omega: Whether to return connectivity matrix
        
        Returns:
            output: [batch, seq_len, d_model]
            omega_sum: [batch, seq_len] (if return_omega=True)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: [batch, n_heads, seq_len, d_k]
        
        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch, n_heads, seq_len, seq_len]
        
        # Compute Hebbian bonus
        hebbian_bonus = self.compute_hebbian_bonus(x12, seq_len)
        # hebbian_bonus: [batch, seq_len, seq_len]
        
        # Add Hebbian modulation (broadcast across heads)
        scores = scores + (self.beta * self.beta_scale) * hebbian_bonus.unsqueeze(1)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        # out: [batch, n_heads, seq_len, d_k]
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        out = self.W_O(out)
        
        if return_omega:
            # Compute total connectivity per token (sum over all attention weights)
            # This is used to update x12
            omega_sum = attn_weights.mean(dim=1).sum(dim=-1)  # [batch, seq_len]
            return out, omega_sum
        
        return out

# ===================================================================
# φ-HARMONIC FEED-FORWARD NETWORK
# ===================================================================

class PhiHarmonicFFN(nn.Module):
    """
    Feed-forward network with φ-scaled hidden dimension.
    
    d_ff = ⌊d_model × φ⌋
    
    Uses GELU activation for smooth gradients.
    """
    
    def __init__(self, config: CosmicConfig) -> None:
        super().__init__()
        self.W1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.W2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        h = self.W1(x)
        h = self.activation(h)
        h = self.dropout(h)
        out = self.W2(h)
        out = self.dropout(out)
        return out

# ===================================================================
# MEMORY-AUGMENTED LAYER
# ===================================================================

class EpisodicMemory(nn.Module):
    """
    Memory buffer that stores past (embedding, x12) pairs.
    Enables learning from history without explicit external memory.
    """
    
    def __init__(self, config: CosmicConfig) -> None:
        super().__init__()
        self.memory_size = config.memory_size
        self.d_model = config.d_model
        self.alpha = config.alpha_memory

        # Memory buffers (non-trainable)
        self.register_buffer(
            'memory_embeddings',
            torch.zeros(config.memory_size, config.d_model)
        )
        self.register_buffer(
            'memory_x12',
            torch.zeros(config.memory_size)
        )
        self.memory_ptr = 0
        self.memory_filled = 0

    def update_memory(self, embeddings: torch.Tensor, x12: torch.Tensor) -> None:
        """Add current states to memory buffer (during training)"""
        if not self.training:
            return

        batch_size = embeddings.shape[0]
        seq_len = embeddings.shape[1]

        # Flatten batch dimension
        embeddings_flat = embeddings.view(-1, self.d_model)
        x12_flat = x12.view(-1)

        # Add to memory (circular buffer)
        for i in range(min(len(embeddings_flat), self.memory_size)):
            self.memory_embeddings[self.memory_ptr] = embeddings_flat[i].detach()
            self.memory_x12[self.memory_ptr] = x12_flat[i].detach()
            self.memory_ptr = (self.memory_ptr + 1) % self.memory_size
            self.memory_filled = min(self.memory_filled + 1, self.memory_size)

    def retrieve(self, query_embeddings: torch.Tensor, query_x12: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory based on similarity.
        
        Args:
            query_embeddings: [batch, seq_len, d_model]
            query_x12: [batch, seq_len]
        
        Returns:
            retrieved: [batch, seq_len, d_model]
        """
        if self.memory_filled == 0:
            return torch.zeros_like(query_embeddings)
        
        batch_size, seq_len, d_model = query_embeddings.shape
        
        # Normalize
        query_norm = F.normalize(query_embeddings, dim=-1)
        mem_norm = F.normalize(self.memory_embeddings[:self.memory_filled], dim=-1)
        
        # Semantic similarity
        # [batch, seq_len, d_model] @ [d_model, memory_filled]
        sim_semantic = torch.matmul(
            query_norm.view(-1, d_model),
            mem_norm.t()
        ).view(batch_size, seq_len, -1)
        
        # Adaptive similarity based on x12
        query_x12_exp = query_x12.unsqueeze(2)  # [batch, seq_len, 1]
        mem_x12_exp = self.memory_x12[:self.memory_filled].unsqueeze(0).unsqueeze(0)
        x12_diff = query_x12_exp - mem_x12_exp
        sim_adaptive = torch.exp(-x12_diff ** 2 / 2)
        
        # Combined similarity
        sim_total = sim_semantic * sim_adaptive
        sim_weights = F.softmax(sim_total, dim=-1)
        
        # Retrieve
        retrieved = torch.matmul(
            sim_weights,
            self.memory_embeddings[:self.memory_filled].unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        return retrieved

# ===================================================================
# COMPLETE 12D CST TRANSFORMER LAYER
# ===================================================================

class CosmicSynapseLayer(nn.Module):
    """
    One complete layer of the 12D Cosmic Synapse Transformer.
    
    Implements:
    1. Hebbian Multi-Head Attention
    2. Internal State Update (x₁₂ dynamics)
    3. φ-Harmonic Feed-Forward
    4. Memory Retrieval
    5. Chaos Injection (during training)
    """
    
    def __init__(self, config: CosmicConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Components
        self.attention = HebbianMultiHeadAttention(config)
        self.ffn = PhiHarmonicFFN(config)
        self.state_dynamics = InternalStateDynamics(config)
        self.memory = EpisodicMemory(config)

        # Layer norms
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        # Chaos generator (one per layer for diversity)
        self.lorenz = LorenzAttractor()

    def inject_chaos(self, x: torch.Tensor) -> torch.Tensor:
        """Inject Lorenz chaos during training"""
        if self.training and torch.rand(1).item() < self.config.p_chaos:
            chaos_noise = self.lorenz.get_noise(x.shape, device=x.device)
            x = x + self.config.lambda_chaos * chaos_noise
        return x

    def forward(self, x: torch.Tensor, x12: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            x12: [batch, seq_len]
            mask: [batch, seq_len, seq_len]
        
        Returns:
            x_out: [batch, seq_len, d_model]
            x12_out: [batch, seq_len]
        """
        # 1. Hebbian Self-Attention with connectivity tracking
        attn_out, omega = self.attention(
            self.ln1(x), x12, mask, return_omega=True
        )
        x = x + attn_out
        
        # 2. Update Internal States based on connectivity
        x12_new = self.state_dynamics(x12, omega)
        
        # 3. Memory Retrieval
        if self.training:
            self.memory.update_memory(x, x12_new)
        
        retrieved = self.memory.retrieve(x, x12_new)
        x = x + self.config.alpha_memory * retrieved
        
        # 4. Feed-Forward
        ff_out = self.ffn(self.ln2(x))
        
        # 5. Chaos Injection
        ff_out = self.inject_chaos(ff_out)
        
        # 6. Residual
        x = x + ff_out
        
        return x, x12_new

# ===================================================================
# COMPLETE 12D COSMIC SYNAPSE TRANSFORMER
# ===================================================================

class CosmicSynapseTransformer(nn.Module):
    """
    Full 12D Cosmic Synapse Transformer implementing the complete theory.
    
    This is a production-grade model that can be trained on any text corpus.
    All mathematical principles from the 12D CST theory are implemented.
    """
    
    def __init__(self, config: CosmicConfig) -> None:
        super().__init__()
        self.config = config

        # Token + Position Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Stack of Cosmic Synapse Layers
        self.layers = nn.ModuleList([
            CosmicSynapseLayer(config, i) for i in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights (standard practice)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize parameters
        self.apply(self._init_weights)

        # Special scaled init for residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith('W_O.weight') or pn.endswith('W2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

        print(f"[12D CST] Model initialized with {self.get_num_params()/1e6:.2f}M parameters")
        print(f"[12D CST] φ-optimized dimensions: d_model={config.d_model}, d_ff={config.d_ff}")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with φ-scaled variance"""
        if isinstance(module, nn.Linear):
            # Use φ for variance scaling
            std = 0.02 * PHI_INV
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            idx: Token indices [batch, seq_len]
            targets: Target indices [batch, seq_len] (for training)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar (if targets provided)
            metrics: dict with x12, omega, etc.
        """
        device = idx.device
        batch_size, seq_len = idx.shape
        
        assert seq_len <= self.config.max_seq_len, \
            f"Sequence length {seq_len} exceeds maximum {self.config.max_seq_len}"
        
        # Get embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.token_embedding(idx)  # [batch, seq_len, d_model]
        pos_emb = self.position_embedding(pos)  # [1, seq_len, d_model]
        x = self.dropout(tok_emb + pos_emb)
        
        # Initialize internal states (x₁₂)
        x12 = torch.zeros(batch_size, seq_len, device=device)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(
            1, seq_len, seq_len
        )
        
        # Pass through all layers
        x12_history = []
        for layer in self.layers:
            x, x12 = layer(x, x12, mask)
            x12_history.append(x12.detach().mean().item())
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        # Metrics
        metrics = {
            'x12_final': x12.mean().item(),
            'x12_std': x12.std().item(),
            'x12_history': x12_history
        }
        
        return logits, loss, metrics
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            idx: Context tokens [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Generated token indices [batch, seq_len + max_new_tokens]
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

# ===================================================================
# TRAINING UTILITIES
# ===================================================================

class CosmicTrainer:
    """Training utilities for 12D CST Transformer"""
    
    def __init__(self, model: CosmicSynapseTransformer, config: CosmicConfig, device: str = 'cuda') -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer with φ-scaled learning rate
        self.lr = 3e-4 * PHI_INV
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        print(f"[TRAINER] Initialized with lr={self.lr:.6f} (φ-scaled)")

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, dict]:
        """Single training step"""
        self.model.train()

        # Unpack batch
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        logits, loss, metrics = self.model(inputs, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Update
        self.optimizer.step()

        return loss.item(), metrics

    @torch.no_grad()
    def validate(self, val_loader) -> float:
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            logits, loss, _ = self.model(inputs, targets)
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches

# ===================================================================
# EXAMPLE USAGE
# ===================================================================

if __name__ == "__main__":
    # Create configuration
    config = CosmicConfig(
        vocab_size=50257,
        max_seq_len=1024,
        d_model=768,
        n_layers=12,
        n_heads=12
    )
    
    # Initialize model
    model = CosmicSynapseTransformer(config)
    
    # Example forward pass
    batch_size = 2
    seq_len = 64
    
    # Random tokens
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward
    logits, loss, metrics = model(idx, targets=idx)
    
    print(f"\nExample Forward Pass:")
    print(f"  Input shape: {idx.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  x₁₂ final mean: {metrics['x12_final']:.4f}")
    print(f"  x₁₂ std: {metrics['x12_std']:.4f}")
    
    # Example generation
    context = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(context, max_new_tokens=50, temperature=0.8, top_k=50)
    
    print(f"\nGeneration:")
    print(f"  Context length: {context.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print(f"\n[12D CST Transformer Ready]")
