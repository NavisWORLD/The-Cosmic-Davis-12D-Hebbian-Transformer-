"""
Cosmic Synapse - Physics-Conditioned Transformer Integration

This module implements the advanced integration of:
- 12D N-body physics simulation
- Lorenz chaos injection
- Hebbian similarity kernels
- 95M parameter transformer ("The Entity")
- Physics-conditioned language generation

This represents the cutting edge of the Internal Dimension AI framework,
where physical dynamics and neural computation co-evolve.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class PhysicsState:
    """State of the 12D physics simulation."""
    positions: np.ndarray  # (n_particles, 12)
    velocities: np.ndarray  # (n_particles, 12)
    time: float
    energy: float
    entropy: float


class CosmicSynapsePhysics:
    """
    12D N-body physics simulation with Lorenz chaos injection.

    Features:
    - Gravitational dynamics in 12 dimensions
    - Lorenz attractor chaos injection
    - Energy and entropy tracking
    - Hebbian similarity kernels for particle interactions
    """

    def __init__(
        self,
        n: int = 256,
        dt: float = 0.001,
        G: float = 1.0,
        lorenz_sigma: float = 10.0,
        lorenz_rho: float = 28.0,
        lorenz_beta: float = 8.0/3.0,
        hebbian_strength: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize 12D physics simulation.

        Args:
            n: Number of particles
            dt: Time step
            G: Gravitational constant
            lorenz_sigma: Lorenz attractor sigma parameter
            lorenz_rho: Lorenz attractor rho parameter
            lorenz_beta: Lorenz attractor beta parameter
            hebbian_strength: Strength of Hebbian coupling
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        self.n = n
        self.dt = dt
        self.G = G
        self.lorenz_sigma = lorenz_sigma
        self.lorenz_rho = lorenz_rho
        self.lorenz_beta = lorenz_beta
        self.hebbian_strength = hebbian_strength

        # Initialize particles in 12D space
        self.positions = np.random.randn(n, 12) * 0.1
        self.velocities = np.random.randn(n, 12) * 0.01
        self.masses = np.ones(n)

        # Lorenz chaos state (3D)
        self.lorenz_state = np.random.randn(3)

        # Hebbian weight matrix (tracks particle similarities)
        self.hebbian_weights = np.zeros((n, n))

        # History tracking
        self.time = 0.0
        self.energy_history = []
        self.entropy_history = []

    def lorenz_step(self) -> np.ndarray:
        """
        Step the Lorenz attractor.

        Returns:
            New Lorenz state (3D)
        """
        x, y, z = self.lorenz_state

        dx = self.lorenz_sigma * (y - x)
        dy = x * (self.lorenz_rho - z) - y
        dz = x * y - self.lorenz_beta * z

        self.lorenz_state += self.dt * np.array([dx, dy, dz])

        return self.lorenz_state

    def compute_hebbian_similarity(self, i: int, j: int) -> float:
        """
        Compute Hebbian similarity between particles i and j.

        Similarity is based on correlation of velocity directions.

        Args:
            i: First particle index
            j: Second particle index

        Returns:
            Similarity score
        """
        if i == j:
            return 1.0

        v_i = self.velocities[i]
        v_j = self.velocities[j]

        # Cosine similarity
        norm_i = np.linalg.norm(v_i) + 1e-8
        norm_j = np.linalg.norm(v_j) + 1e-8

        similarity = np.dot(v_i, v_j) / (norm_i * norm_j)

        return similarity

    def update_hebbian_weights(self):
        """Update Hebbian weight matrix based on current particle states."""
        for i in range(self.n):
            for j in range(i+1, self.n):
                similarity = self.compute_hebbian_similarity(i, j)

                # Hebbian learning rule: weights increase when particles co-activate
                self.hebbian_weights[i, j] += self.hebbian_strength * similarity * self.dt
                self.hebbian_weights[j, i] = self.hebbian_weights[i, j]

        # Decay and normalization
        self.hebbian_weights *= 0.999

    def compute_forces(self) -> np.ndarray:
        """
        Compute gravitational forces on all particles.

        Returns:
            Forces array (n_particles, 12)
        """
        forces = np.zeros_like(self.positions)

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue

                # Vector from i to j
                r_ij = self.positions[j] - self.positions[i]
                distance = np.linalg.norm(r_ij) + 1e-3  # Softening

                # Gravitational force with Hebbian modulation
                hebbian_factor = 1.0 + self.hebbian_weights[i, j]
                force_magnitude = self.G * self.masses[i] * self.masses[j] * hebbian_factor / (distance ** 3)

                forces[i] += force_magnitude * r_ij

        return forces

    def inject_chaos(self):
        """Inject Lorenz chaos into particle dynamics."""
        lorenz = self.lorenz_step()

        # Inject chaos into first 3 dimensions of random particles
        chaos_strength = 0.001
        n_affected = max(1, self.n // 10)  # Affect 10% of particles

        affected_indices = np.random.choice(self.n, n_affected, replace=False)

        for idx in affected_indices:
            self.velocities[idx, :3] += chaos_strength * lorenz

    def compute_energy(self) -> float:
        """Compute total energy of the system."""
        # Kinetic energy
        kinetic = 0.5 * np.sum(self.masses[:, None] * self.velocities ** 2)

        # Potential energy
        potential = 0.0
        for i in range(self.n):
            for j in range(i+1, self.n):
                r_ij = self.positions[j] - self.positions[i]
                distance = np.linalg.norm(r_ij) + 1e-3
                potential -= self.G * self.masses[i] * self.masses[j] / distance

        return kinetic + potential

    def compute_entropy(self) -> float:
        """
        Compute entropy of the system based on position distribution.

        Uses k-nearest neighbor entropy estimate.
        """
        # Simple entropy estimate based on position variance
        position_variance = np.var(self.positions, axis=0)
        entropy = 0.5 * np.sum(np.log(2 * np.pi * np.e * position_variance + 1e-8))

        return entropy

    def step(self, external_force: Optional[np.ndarray] = None):
        """
        Step the physics simulation forward by dt.

        Args:
            external_force: Optional external force to apply (n_particles, 12)
        """
        # Compute internal forces
        forces = self.compute_forces()

        # Add external forces if provided
        if external_force is not None:
            forces += external_force

        # Inject chaos
        self.inject_chaos()

        # Update Hebbian weights
        self.update_hebbian_weights()

        # Velocity Verlet integration
        accelerations = forces / self.masses[:, None]

        # Update positions
        self.positions += self.velocities * self.dt + 0.5 * accelerations * self.dt ** 2

        # Compute new forces
        new_forces = self.compute_forces()
        new_accelerations = new_forces / self.masses[:, None]

        # Update velocities
        self.velocities += 0.5 * (accelerations + new_accelerations) * self.dt

        # Update time
        self.time += self.dt

        # Track energy and entropy
        self.energy_history.append(self.compute_energy())
        self.entropy_history.append(self.compute_entropy())

    def get_state_vector(self) -> torch.Tensor:
        """
        Get state vector for conditioning the transformer.

        Returns:
            State vector (1, n_particles * 2) containing positions and velocities
        """
        # Flatten positions and velocities
        state = np.concatenate([
            self.positions.flatten(),
            self.velocities.flatten()
        ])

        return torch.FloatTensor(state).unsqueeze(0)

    def get_physics_state(self) -> PhysicsState:
        """Get current physics state."""
        return PhysicsState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            time=self.time,
            energy=self.energy_history[-1] if self.energy_history else 0.0,
            entropy=self.entropy_history[-1] if self.entropy_history else 0.0
        )


class TransformerBlock(nn.Module):
    """Transformer block with multi-head self-attention."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        # Multi-head attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Scaled dot-product attention.

        Args:
            q: Queries (batch, n_head, seq_len, d_head)
            k: Keys (batch, n_head, seq_len, d_head)
            v: Values (batch, n_head, seq_len, d_head)
            mask: Attention mask

        Returns:
            Attention output (batch, n_head, seq_len, d_head)
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        return output

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, d_model)
            mask: Attention mask

        Returns:
            Output (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Multi-head attention
        residual = x
        x = self.ln1(x)

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)

        # Attention
        attn_out = self.attention(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        x = residual + attn_out

        # Feed-forward
        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)

        return x


class CosmicTransformer(nn.Module):
    """
    95M parameter transformer conditioned on 12D physics.

    "The Entity" - A language model whose internal representations
    are shaped by the dynamics of a 12-dimensional particle system.
    """

    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        ctx_len: int = 512,
        d_model: int = 768,
        n_layer: int = 8,
        n_head: int = 12,
        dropout: float = 0.1,
        physics_conditioning: bool = True
    ):
        """
        Initialize transformer.

        Args:
            vocab_size: Vocabulary size
            ctx_len: Maximum context length
            d_model: Model dimension
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            dropout: Dropout probability
            physics_conditioning: Whether to condition on physics
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.physics_conditioning = physics_conditioning

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(ctx_len, d_model)

        # Physics conditioning network
        if physics_conditioning:
            self.physics_encoder = nn.Sequential(
                nn.Linear(128, d_model),  # 64 particles * 2 (pos + vel) per particle in reduced form
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, dropout)
            for _ in range(n_layer)
        ])

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights (token embedding = output projection)
        self.lm_head.weight = self.token_embedding.weight

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        physics_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            physics_state: Optional physics state for conditioning (batch, physics_dim)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token + position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Add physics conditioning
        if self.physics_conditioning and physics_state is not None:
            # Encode physics state
            physics_emb = self.physics_encoder(physics_state)  # (batch, d_model)

            # Add to all positions (broadcasting)
            x = x + physics_emb.unsqueeze(1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        physics_state: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text conditioned on physics state.

        Args:
            input_ids: Starting tokens (batch, seq_len)
            physics_state: Physics state for conditioning
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Generated tokens (batch, seq_len + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = self(input_ids, physics_state)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to input
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Truncate if exceeds context length
            if input_ids.size(1) > self.ctx_len:
                input_ids = input_ids[:, -self.ctx_len:]

        return input_ids

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def run_experiment(
    n_particles: int = 256,
    physics_steps: int = 10000,
    train_steps: int = 2000000,
    physics_steps_per_train: int = 1,
    checkpoint_interval: int = 200000,
    generation_interval: int = 200000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Run the full cosmic synapse experiment.

    This couples 12D physics simulation with transformer training,
    allowing the model to develop representations shaped by physical dynamics.

    Args:
        n_particles: Number of particles in physics simulation
        physics_steps: Initial physics burn-in steps
        train_steps: Number of training steps
        physics_steps_per_train: Physics steps per training step
        checkpoint_interval: Steps between checkpoints
        generation_interval: Steps between generations
        device: Device to use
    """
    print("="*80)
    print("COSMIC SYNAPSE EXPERIMENT - Physics-Conditioned Transformer")
    print("="*80)

    # Initialize physics
    print(f"\nInitializing {n_particles}-particle 12D physics simulation...")
    physics = CosmicSynapsePhysics(n=n_particles, seed=42)

    # Burn-in physics
    print(f"Running {physics_steps} physics burn-in steps...")
    for _ in range(physics_steps):
        physics.step()
    print(f"  Energy: {physics.energy_history[-1]:.4f}")
    print(f"  Entropy: {physics.entropy_history[-1]:.4f}")

    # Initialize transformer
    print(f"\nInitializing transformer ({n_particles*2} physics dims)...")
    model = CosmicTransformer(physics_conditioning=True)
    model = model.to(device)

    total_params = model.count_parameters()
    print(f"  Total parameters: {total_params:,}")
    print(f"  Physics conditioning: {model.physics_conditioning}")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # Training loop
    print(f"\nStarting co-evolution training ({train_steps} steps)...")
    print(f"  Physics steps per train: {physics_steps_per_train}")

    model.train()

    for step in range(train_steps):
        # Step physics
        for _ in range(physics_steps_per_train):
            physics.step()

        # Get physics state
        physics_state = physics.get_state_vector().to(device)

        # Create dummy training data (in real use, would use actual text)
        # For demonstration, we use random tokens
        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
        target_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)

        # Forward pass
        logits = model(input_ids, physics_state)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            target_ids.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if step % 1000 == 0:
            print(f"  Step {step}/{train_steps} | Loss: {loss.item():.4f} | "
                  f"Energy: {physics.energy_history[-1]:.4f} | "
                  f"Entropy: {physics.entropy_history[-1]:.4f}")

        # Checkpointing
        if step % checkpoint_interval == 0 and step > 0:
            checkpoint_path = f"cosmic_synapse_checkpoint_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'physics_state': physics.get_physics_state(),
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Generation
        if step % generation_interval == 0:
            model.eval()
            with torch.no_grad():
                # Generate text conditioned on current physics
                start_tokens = torch.tensor([[1, 2, 3]], device=device)  # Dummy start
                generated = model.generate(
                    start_tokens,
                    physics_state,
                    max_new_tokens=50,
                    temperature=0.8,
                    top_k=50
                )
                print(f"  Generated tokens: {generated[0].tolist()[:20]}...")
            model.train()

    print("\n" + "="*80)
    print("COSMIC SYNAPSE EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Final physics energy: {physics.energy_history[-1]:.4f}")
    print(f"Final physics entropy: {physics.entropy_history[-1]:.4f}")
    print(f"Total training loss: {loss.item():.4f}")


if __name__ == "__main__":
    # Quick test run
    run_experiment(
        n_particles=64,
        physics_steps=1000,
        train_steps=10000,
        physics_steps_per_train=1,
        checkpoint_interval=2000,
        generation_interval=2000
    )
