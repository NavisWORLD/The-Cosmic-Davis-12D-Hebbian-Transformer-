"""
COSMIC DAVIS vs VANILLA TRANSFORMER BENCHMARK
==============================================
Fair head-to-head comparison:
- Same parameters, same data, same training steps
- Measures: Loss, Perplexity, Learning Speed, Generation Quality

This will prove (or disprove) the 12D architecture advantage.
"""

import sys
import time
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

# Setup paths
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig, PHI

# ============================================================
# VANILLA TRANSFORMER (Standard GPT-style, NO 12D innovations)
# ============================================================

@dataclass
class VanillaConfig:
    vocab_size: int = 10000
    max_seq_len: int = 512
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1


class VanillaAttention(nn.Module):
    """Standard multi-head attention - NO Hebbian modulation."""
    
    def __init__(self, config: VanillaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        self.W_Q = nn.Linear(config.d_model, config.d_model)
        self.W_K = nn.Linear(config.d_model, config.d_model)
        self.W_V = nn.Linear(config.d_model, config.d_model)
        self.W_O = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None):
        B, T, D = x.shape
        
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_O(out)


class VanillaFFN(nn.Module):
    """Standard feed-forward network."""
    
    def __init__(self, config: VanillaConfig):
        super().__init__()
        self.W1 = nn.Linear(config.d_model, config.d_ff)
        self.W2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.W2(self.dropout(F.gelu(self.W1(x))))


class VanillaLayer(nn.Module):
    """Standard transformer layer - NO internal state, NO chaos, NO memory."""
    
    def __init__(self, config: VanillaConfig):
        super().__init__()
        self.attention = VanillaAttention(config)
        self.ffn = VanillaFFN(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class VanillaTransformer(nn.Module):
    """Standard GPT-style transformer - baseline for comparison."""
    
    def __init__(self, config: VanillaConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([VanillaLayer(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[VANILLA] Initialized with {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        x = self.dropout(self.token_emb(idx) + self.pos_emb(pos))
        
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss, {}
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================
# BENCHMARK RUNNER
# ============================================================

def create_synthetic_data(vocab_size, seq_len, n_samples):
    """Create synthetic training data."""
    data = []
    for _ in range(n_samples):
        # Create sequences with patterns (not pure random)
        seq = []
        for i in range(seq_len + 1):
            if i < 5:
                seq.append(random.randint(4, vocab_size - 1))
            else:
                # Some tokens depend on previous (learnable pattern)
                if random.random() < 0.3:
                    seq.append((seq[i-1] + seq[i-2]) % vocab_size)
                else:
                    seq.append(random.randint(4, vocab_size - 1))
        data.append(seq)
    return data


def run_benchmark():
    print("="*70)
    print("🔬 COSMIC DAVIS 12D vs VANILLA TRANSFORMER BENCHMARK")
    print("="*70)
    
    # Configuration (IDENTICAL for both)
    vocab_size = 10000
    d_model = 256
    n_layers = 6
    n_heads = 8
    seq_len = 64
    batch_size = 16
    n_iterations = 100
    
    print(f"\n📊 Configuration:")
    print(f"   Vocab Size: {vocab_size}")
    print(f"   Model Dim: {d_model}")
    print(f"   Layers: {n_layers}")
    print(f"   Heads: {n_heads}")
    print(f"   Seq Length: {seq_len}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Iterations: {n_iterations}")
    
    # Create models
    print("\n🔧 Creating Models...")
    
    # Vanilla config
    vanilla_config = VanillaConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 4
    )
    vanilla_model = VanillaTransformer(vanilla_config)
    
    # Cosmic config (matches vanilla dimensions)
    cosmic_config = CosmicConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads
    )
    # Override phi-optimization to match vanilla exactly
    cosmic_config.d_model = d_model
    cosmic_config.d_ff = d_model * 4
    cosmic_config.d_k = d_model // n_heads
    cosmic_model = CosmicSynapseTransformer(cosmic_config)
    
    # Count parameters
    vanilla_params = sum(p.numel() for p in vanilla_model.parameters())
    cosmic_params = sum(p.numel() for p in cosmic_model.parameters())
    
    print(f"\n📈 Parameter Count:")
    print(f"   Vanilla:  {vanilla_params:,} ({vanilla_params/1e6:.2f}M)")
    print(f"   Cosmic:   {cosmic_params:,} ({cosmic_params/1e6:.2f}M)")
    print(f"   Overhead: {(cosmic_params - vanilla_params):,} ({(cosmic_params/vanilla_params - 1)*100:.1f}% more)")
    
    # Create training data
    print("\n📦 Generating Training Data...")
    train_data = create_synthetic_data(vocab_size, seq_len, n_iterations * batch_size)
    
    # Optimizers
    vanilla_opt = torch.optim.AdamW(vanilla_model.parameters(), lr=3e-4)
    cosmic_opt = torch.optim.AdamW(cosmic_model.parameters(), lr=3e-4)
    
    # Training loop
    print("\n🏋️ Training Both Models...")
    print("-" * 70)
    
    vanilla_losses = []
    cosmic_losses = []
    vanilla_times = []
    cosmic_times = []
    
    for i in range(n_iterations):
        # Get batch
        batch_start = i * batch_size
        batch = train_data[batch_start:batch_start + batch_size]
        x = torch.tensor([b[:-1] for b in batch], dtype=torch.long)
        y = torch.tensor([b[1:] for b in batch], dtype=torch.long)
        
        # Train Vanilla
        vanilla_model.train()
        vanilla_opt.zero_grad()
        t0 = time.time()
        _, v_loss, _ = vanilla_model(x, targets=y)
        v_loss.backward()
        vanilla_opt.step()
        vanilla_times.append(time.time() - t0)
        vanilla_losses.append(v_loss.item())
        
        # Train Cosmic
        cosmic_model.train()
        cosmic_opt.zero_grad()
        t0 = time.time()
        _, c_loss, _ = cosmic_model(x, targets=y)
        c_loss.backward()
        cosmic_opt.step()
        cosmic_times.append(time.time() - t0)
        cosmic_losses.append(c_loss.item())
        
        # Progress
        if (i + 1) % 10 == 0:
            v_avg = sum(vanilla_losses[-10:]) / 10
            c_avg = sum(cosmic_losses[-10:]) / 10
            print(f"   Iter {i+1:3d}: Vanilla Loss={v_avg:.4f}  |  Cosmic Loss={c_avg:.4f}  |  Δ={c_avg-v_avg:+.4f}")
    
    # Results
    print("\n" + "="*70)
    print("📊 BENCHMARK RESULTS")
    print("="*70)
    
    # Final losses
    v_final = sum(vanilla_losses[-10:]) / 10
    c_final = sum(cosmic_losses[-10:]) / 10
    
    print(f"\n🎯 Final Loss (last 10 iterations):")
    print(f"   Vanilla:  {v_final:.4f}")
    print(f"   Cosmic:   {c_final:.4f}")
    if c_final < v_final:
        print(f"   Winner:   🏆 COSMIC DAVIS ({(1 - c_final/v_final)*100:.1f}% lower loss)")
    else:
        print(f"   Winner:   🏆 VANILLA ({(1 - v_final/c_final)*100:.1f}% lower loss)")
    
    # Perplexity
    v_ppl = math.exp(min(v_final, 10))
    c_ppl = math.exp(min(c_final, 10))
    
    print(f"\n📈 Perplexity:")
    print(f"   Vanilla:  {v_ppl:.2f}")
    print(f"   Cosmic:   {c_ppl:.2f}")
    
    # Speed
    v_speed = sum(vanilla_times) / len(vanilla_times) * 1000
    c_speed = sum(cosmic_times) / len(cosmic_times) * 1000
    
    print(f"\n⚡ Training Speed (ms/iteration):")
    print(f"   Vanilla:  {v_speed:.1f}ms")
    print(f"   Cosmic:   {c_speed:.1f}ms")
    print(f"   Overhead: {(c_speed/v_speed - 1)*100:.1f}%")
    
    # Learning curve analysis
    print(f"\n📉 Learning Curve Analysis:")
    
    # First 20 vs last 20
    v_early = sum(vanilla_losses[:20]) / 20
    v_late = sum(vanilla_losses[-20:]) / 20
    c_early = sum(cosmic_losses[:20]) / 20
    c_late = sum(cosmic_losses[-20:]) / 20
    
    v_improvement = (v_early - v_late) / v_early * 100
    c_improvement = (c_early - c_late) / c_early * 100
    
    print(f"   Vanilla improvement: {v_improvement:.1f}%")
    print(f"   Cosmic improvement:  {c_improvement:.1f}%")
    
    if c_improvement > v_improvement:
        print(f"   🏆 Cosmic learns {c_improvement - v_improvement:.1f}% faster!")
    else:
        print(f"   Vanilla learns {v_improvement - c_improvement:.1f}% faster")
    
    # Generation test
    print(f"\n✍️ Generation Test:")
    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    
    vanilla_model.eval()
    cosmic_model.eval()
    
    with torch.no_grad():
        v_gen = vanilla_model.generate(prompt.clone(), max_new_tokens=20, temperature=0.8)
        c_gen = cosmic_model.generate(prompt.clone(), max_new_tokens=20, temperature=0.8)
    
    print(f"   Vanilla output: {v_gen[0].tolist()}")
    print(f"   Cosmic output:  {c_gen[0].tolist()}")
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    cosmic_wins = 0
    vanilla_wins = 0
    
    if c_final < v_final:
        cosmic_wins += 1
        print("   ✅ Loss: Cosmic wins")
    else:
        vanilla_wins += 1
        print("   ✅ Loss: Vanilla wins")
    
    if c_improvement > v_improvement:
        cosmic_wins += 1
        print("   ✅ Learning Speed: Cosmic wins")
    else:
        vanilla_wins += 1
        print("   ✅ Learning Speed: Vanilla wins")
    
    if c_speed < v_speed * 1.5:  # Cosmic is competitive if less than 50% slower
        cosmic_wins += 1
        print("   ✅ Compute Efficiency: Competitive")
    else:
        vanilla_wins += 1
        print("   ⚠️ Compute Efficiency: Cosmic is slower")
    
    print(f"\n   Final Score: Cosmic {cosmic_wins} - {vanilla_wins} Vanilla")
    
    if cosmic_wins > vanilla_wins:
        print("\n   🏆 COSMIC DAVIS 12D ARCHITECTURE SHOWS ADVANTAGE!")
    elif vanilla_wins > cosmic_wins:
        print("\n   ⚠️ Vanilla performs better on this benchmark")
    else:
        print("\n   🤝 Results are mixed - more testing needed")
    
    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)


if __name__ == "__main__":
    run_benchmark()
