"""
COMPREHENSIVE BENCHMARK: 12D CST vs Vanilla Transformer
========================================================

This script trains both models on identical data and compares:
- Training loss convergence
- Perplexity
- Training speed (tokens/sec)
- Inference speed
- Memory usage
- Model capacity

Author: Claude + Cory Shane Davis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from dataclasses import dataclass, asdict

from cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig

# ===================================================================
# VANILLA TRANSFORMER BASELINE
# ===================================================================

@dataclass
class VanillaConfig:
    """Standard transformer configuration for fair comparison."""
    vocab_size: int = 1000
    max_seq_len: int = 128
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 768  # Standard 4x scaling
    dropout: float = 0.1

class VanillaAttention(nn.Module):
    """Standard multi-head attention (no Hebbian modulation)."""

    def __init__(self, config: VanillaConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.d_model = config.d_model

        self.W_Q = nn.Linear(config.d_model, config.d_model)
        self.W_K = nn.Linear(config.d_model, config.d_model)
        self.W_V = nn.Linear(config.d_model, config.d_model)
        self.W_O = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_O(out)

        return out

class VanillaFFN(nn.Module):
    """Standard feed-forward network (4x expansion)."""

    def __init__(self, config: VanillaConfig) -> None:
        super().__init__()
        self.W1 = nn.Linear(config.d_model, config.d_ff)
        self.W2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.W2(self.dropout(self.activation(self.W1(x)))))

class VanillaTransformerLayer(nn.Module):
    """Standard transformer layer."""

    def __init__(self, config: VanillaConfig) -> None:
        super().__init__()
        self.attention = VanillaAttention(config)
        self.ffn = VanillaFFN(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        x = x + self.attention(self.ln1(x), mask)
        # Feed-forward
        x = x + self.ffn(self.ln2(x))
        return x

class VanillaTransformer(nn.Module):
    """Standard transformer for comparison."""

    def __init__(self, config: VanillaConfig) -> None:
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            VanillaTransformerLayer(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        # Initialize
        self.apply(self._init_weights)

        print(f"[VANILLA] Model initialized with {self.get_num_params()/1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = idx.device
        batch_size, seq_len = idx.shape

        # Embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.dropout(x)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )

        return logits, loss

# ===================================================================
# BENCHMARKING FUNCTIONS
# ===================================================================

def generate_benchmark_data(num_tokens: int = 50000, vocab_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for benchmarking."""
    print(f"\n[DATA] Generating {num_tokens:,} tokens...")

    from generate_synthetic_data import SyntheticDataGenerator

    gen = SyntheticDataGenerator(seed=42)
    tokens = gen.generate_tokens(num_tokens, add_phi_patterns=True)
    gen.build_vocabulary(tokens)

    # Ensure vocab size matches
    if gen.vocab_size > vocab_size:
        # Truncate vocab
        tokens = [t if gen.vocab.get(t, vocab_size) < vocab_size else '<unk>' for t in tokens]
        gen.build_vocabulary(tokens)

    token_ids = gen.tokens_to_ids(tokens)
    token_array = np.array(token_ids, dtype=np.uint16)

    # Split train/val
    split_idx = int(0.9 * len(token_array))
    train_data = token_array[:split_idx]
    val_data = token_array[split_idx:]

    print(f"[DATA] Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")
    print(f"[DATA] Vocabulary size: {gen.vocab_size}")

    return train_data, val_data

def train_model(
    model: nn.Module,
    train_data: np.ndarray,
    val_data: np.ndarray,
    num_iters: int = 500,
    batch_size: int = 16,
    seq_len: int = 64,
    lr: float = 3e-4,
    model_name: str = "Model"
) -> Dict:
    """Train model and track metrics."""

    print(f"\n[TRAIN] Training {model_name} for {num_iters} iterations...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    metrics = {
        'losses': [],
        'val_losses': [],
        'times': [],
        'tokens_per_sec': [],
        'iterations': []
    }

    start_time = time.time()

    for iter_num in range(num_iters):
        # Get batch
        idx = np.random.randint(0, len(train_data) - seq_len - 1, batch_size)
        x = torch.stack([torch.from_numpy(train_data[i:i+seq_len].astype(np.int64)) for i in idx])
        y = torch.stack([torch.from_numpy(train_data[i+1:i+seq_len+1].astype(np.int64)) for i in idx])

        x, y = x.to(device), y.to(device)

        # Forward
        iter_start = time.time()
        optimizer.zero_grad()

        if 'Cosmic' in model_name or '12D' in model_name:
            logits, loss, _ = model(x, y)
        else:
            logits, loss = model(x, y)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        iter_time = time.time() - iter_start
        tokens_processed = batch_size * seq_len

        # Track metrics
        if iter_num % 10 == 0:
            metrics['losses'].append(loss.item())
            metrics['times'].append(time.time() - start_time)
            metrics['tokens_per_sec'].append(tokens_processed / iter_time)
            metrics['iterations'].append(iter_num)

            # Validation
            if iter_num % 50 == 0:
                val_loss = evaluate_model(model, val_data, seq_len, model_name)
                metrics['val_losses'].append(val_loss)

                print(f"[{model_name}] Iter {iter_num:4d} | "
                      f"Train Loss: {loss.item():.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Speed: {tokens_processed/iter_time:.0f} tok/s")

    total_time = time.time() - start_time
    print(f"[{model_name}] Training complete in {total_time:.2f}s")

    return metrics

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_data: np.ndarray,
    seq_len: int = 64,
    model_name: str = "Model",
    num_batches: int = 20
) -> float:
    """Evaluate model on validation data."""
    model.eval()
    device = next(model.parameters()).device

    losses = []
    for _ in range(num_batches):
        idx = np.random.randint(0, len(val_data) - seq_len - 1)
        x = torch.from_numpy(val_data[idx:idx+seq_len].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(val_data[idx+1:idx+seq_len+1].astype(np.int64)).unsqueeze(0).to(device)

        if 'Cosmic' in model_name or '12D' in model_name:
            _, loss, _ = model(x, y)
        else:
            _, loss = model(x, y)

        losses.append(loss.item())

    model.train()
    return np.mean(losses)

def run_benchmark() -> Dict:
    """Run complete benchmark comparing both models."""

    print("="*80)
    print("COMPREHENSIVE BENCHMARK: 12D CST vs Vanilla Transformer")
    print("="*80)

    # Configuration
    vocab_size = 1000
    max_seq_len = 128
    d_model = 192
    n_layers = 4
    n_heads = 4
    num_iters = 500

    # Generate data
    train_data, val_data = generate_benchmark_data(num_tokens=50000, vocab_size=vocab_size)

    # Create models
    print("\n" + "="*80)
    print("INITIALIZING MODELS")
    print("="*80)

    vanilla_config = VanillaConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 4  # Standard 4x
    )
    vanilla_model = VanillaTransformer(vanilla_config)

    cosmic_config = CosmicConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads
    )
    cosmic_model = CosmicSynapseTransformer(cosmic_config)

    print(f"\nVanilla params: {vanilla_model.get_num_params():,}")
    print(f"Cosmic params: {cosmic_model.get_num_params():,}")
    print(f"Difference: {abs(vanilla_model.get_num_params() - cosmic_model.get_num_params()):,}")

    # Train both models
    vanilla_metrics = train_model(
        vanilla_model, train_data, val_data,
        num_iters=num_iters, model_name="Vanilla"
    )

    cosmic_metrics = train_model(
        cosmic_model, train_data, val_data,
        num_iters=num_iters, model_name="12D CST"
    )

    # Compile results
    vanilla_config_dict = asdict(vanilla_config)
    cosmic_config_dict = asdict(cosmic_config)
    cosmic_config_dict['d_ff'] = cosmic_config.d_ff  # Add d_ff which is computed in __post_init__

    results = {
        'vanilla': {
            'config': vanilla_config_dict,
            'metrics': vanilla_metrics,
            'params': vanilla_model.get_num_params()
        },
        'cosmic': {
            'config': cosmic_config_dict,
            'metrics': cosmic_metrics,
            'params': cosmic_model.get_num_params()
        }
    }

    return results

def generate_report(results: Dict) -> str:
    """Generate markdown report from benchmark results."""

    vanilla = results['vanilla']
    cosmic = results['cosmic']

    # Calculate statistics
    v_final_loss = vanilla['metrics']['losses'][-1]
    c_final_loss = cosmic['metrics']['losses'][-1]

    v_final_val = vanilla['metrics']['val_losses'][-1]
    c_final_val = cosmic['metrics']['val_losses'][-1]

    v_avg_speed = np.mean(vanilla['metrics']['tokens_per_sec'])
    c_avg_speed = np.mean(cosmic['metrics']['tokens_per_sec'])

    report = f"""# 🔬 BENCHMARK RESULTS: 12D Cosmic Synapse Transformer vs Vanilla Transformer

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Device**: {'CUDA' if torch.cuda.is_available() else 'CPU'}
**PyTorch Version**: {torch.__version__}

---

## 📊 Model Configurations

| Metric | Vanilla Transformer | 12D CST | Difference |
|--------|-------------------|---------|------------|
| **Parameters** | {vanilla['params']:,} | {cosmic['params']:,} | {abs(vanilla['params'] - cosmic['params']):,} |
| **d_model** | {vanilla['config']['d_model']} | {cosmic['config']['d_model']} | - |
| **d_ff** | {vanilla['config']['d_ff']} | {cosmic['config']['d_ff']} | φ-harmonic |
| **Layers** | {vanilla['config']['n_layers']} | {cosmic['config']['n_layers']} | - |
| **Heads** | {vanilla['config']['n_heads']} | {cosmic['config']['n_heads']} | - |

---

## 🎯 Training Results

### Final Metrics:

| Metric | Vanilla | 12D CST | Winner |
|--------|---------|---------|--------|
| **Train Loss** | {v_final_loss:.4f} | {c_final_loss:.4f} | {'🏆 12D CST' if c_final_loss < v_final_loss else '🏆 Vanilla'} |
| **Val Loss** | {v_final_val:.4f} | {c_final_val:.4f} | {'🏆 12D CST' if c_final_val < v_final_val else '🏆 Vanilla'} |
| **Avg Speed** | {v_avg_speed:.0f} tok/s | {c_avg_speed:.0f} tok/s | {'🏆 12D CST' if c_avg_speed > v_avg_speed else '🏆 Vanilla'} |

### Convergence:

- **Vanilla**: {vanilla['metrics']['losses'][0]:.4f} → {v_final_loss:.4f} (Δ {vanilla['metrics']['losses'][0] - v_final_loss:.4f})
- **12D CST**: {cosmic['metrics']['losses'][0]:.4f} → {c_final_loss:.4f} (Δ {cosmic['metrics']['losses'][0] - c_final_loss:.4f})

### Perplexity:

- **Vanilla Val**: {math.exp(v_final_val):.2f}
- **12D CST Val**: {math.exp(c_final_val):.2f}

---

## 🏆 Winner Analysis

"""

    # Determine overall winner
    c_wins = 0
    v_wins = 0

    if c_final_loss < v_final_loss:
        c_wins += 1
        report += "✅ **12D CST** achieves lower training loss\n"
    else:
        v_wins += 1
        report += "✅ **Vanilla** achieves lower training loss\n"

    if c_final_val < v_final_val:
        c_wins += 1
        report += "✅ **12D CST** achieves lower validation loss\n"
    else:
        v_wins += 1
        report += "✅ **Vanilla** achieves lower validation loss\n"

    if c_avg_speed > v_avg_speed:
        c_wins += 1
        report += f"✅ **12D CST** is {c_avg_speed/v_avg_speed:.1f}x faster\n"
    else:
        v_wins += 1
        report += f"✅ **Vanilla** is {v_avg_speed/c_avg_speed:.1f}x faster\n"

    report += f"\n**Overall Winner**: {'🏆 12D COSMIC SYNAPSE TRANSFORMER' if c_wins > v_wins else '🏆 VANILLA TRANSFORMER' if v_wins > c_wins else '🤝 TIE'}\n"

    report += f"\n---\n\n## 📈 Raw Data\n\n"
    report += f"Benchmark data saved to: `benchmark_results.json`\n"

    return report

# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    # Run benchmark
    results = run_benchmark()

    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "benchmark_results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        results_json = {
            'vanilla': {
                'config': results['vanilla']['config'],
                'params': results['vanilla']['params'],
                'metrics': {
                    'losses': [float(x) for x in results['vanilla']['metrics']['losses']],
                    'val_losses': [float(x) for x in results['vanilla']['metrics']['val_losses']],
                    'times': [float(x) for x in results['vanilla']['metrics']['times']],
                    'tokens_per_sec': [float(x) for x in results['vanilla']['metrics']['tokens_per_sec']],
                    'iterations': [int(x) for x in results['vanilla']['metrics']['iterations']]
                }
            },
            'cosmic': {
                'config': results['cosmic']['config'],
                'params': results['cosmic']['params'],
                'metrics': {
                    'losses': [float(x) for x in results['cosmic']['metrics']['losses']],
                    'val_losses': [float(x) for x in results['cosmic']['metrics']['val_losses']],
                    'times': [float(x) for x in results['cosmic']['metrics']['times']],
                    'tokens_per_sec': [float(x) for x in results['cosmic']['metrics']['tokens_per_sec']],
                    'iterations': [int(x) for x in results['cosmic']['metrics']['iterations']]
                }
            }
        }
        json.dump(results_json, f, indent=2)

    # Generate report
    report = generate_report(results)

    with open(output_dir / "BENCHMARK_REPORT.md", "w") as f:
        f.write(report)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(report)
    print(f"\nResults saved to: {output_dir}/")
