"""
ULTRA-EXTENDED TRAINING EXPERIMENT
===================================

Push both models to their limits with 10,000 iterations.
Goal: See if 12D CST's advantage continues to grow with extreme training duration.

Based on Experiment 1 results:
- 2000 iterations: 12D CST won by 12.8% (1.1295 vs 1.2957 val loss)
- Hypothesis: Gap will widen further with more training

This experiment will:
1. Train for 10,000 iterations (5x previous)
2. Track detailed convergence curves
3. Save checkpoints every 1000 iterations
4. Analyze learning dynamics over time
"""

import torch
import torch.nn as nn
import time
import math
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass, asdict

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not available - skipping plots")

from cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from benchmark_transformer import (
    VanillaTransformer, VanillaConfig,
    train_model, evaluate_model
)
from generate_synthetic_data import SyntheticDataGenerator

def generate_ultra_dataset(num_tokens: int = 2000000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ultra-large dataset (2M tokens)."""
    print(f"\n[DATA] Generating {num_tokens:,} tokens (ultra-large scale)...")

    start = time.time()
    gen = SyntheticDataGenerator(seed=42)
    tokens = gen.generate_tokens(num_tokens, add_phi_patterns=True)
    gen.build_vocabulary(tokens)
    token_ids = gen.tokens_to_ids(tokens)
    token_array = np.array(token_ids, dtype=np.uint16)

    # Split
    split_idx = int(0.9 * len(token_array))
    train_data = token_array[:split_idx]
    val_data = token_array[split_idx:]

    gen_time = time.time() - start
    print(f"[DATA] Generated in {gen_time:.1f}s ({num_tokens/gen_time:.0f} tok/s)")
    print(f"[DATA] Train: {len(train_data):,} tokens")
    print(f"[DATA] Val: {len(val_data):,} tokens")
    print(f"[DATA] Vocab size: {gen.vocab_size}")

    return train_data, val_data

def train_with_checkpoints(
    model: nn.Module,
    train_data: np.ndarray,
    val_data: np.ndarray,
    num_iters: int = 10000,
    batch_size: int = 16,
    seq_len: int = 64,
    lr: float = 3e-4,
    model_name: str = "Model",
    checkpoint_interval: int = 1000
) -> Dict:
    """Train model with checkpointing and detailed metrics."""

    print(f"\n[TRAIN] Training {model_name} for {num_iters} iterations...")
    print(f"[TRAIN] Checkpoint interval: {checkpoint_interval} iterations")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Metrics tracking
    losses = []
    val_losses = []
    tokens_per_sec = []
    learning_rates = []
    iterations = []

    # Checkpoint directory
    checkpoint_dir = Path("ultra_checkpoints") / model_name.replace(" ", "_")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for iter_num in range(num_iters):
        # Cosine learning rate with φ-modulation
        decay_ratio = iter_num / num_iters
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        phi_factor = 1.0 + 0.1 * math.sin(2 * math.pi * decay_ratio * 1.618033988749895)
        current_lr = 1e-4 + coeff * (lr - 1e-4) * phi_factor

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Sample batch
        idx = np.random.randint(0, len(train_data) - seq_len - 1, size=batch_size)
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

        # Calculate speed
        iter_time = time.time() - iter_start
        speed = (batch_size * seq_len) / iter_time

        # Track metrics
        losses.append(loss.item())
        tokens_per_sec.append(speed)
        learning_rates.append(current_lr)
        iterations.append(iter_num)

        # Evaluate and log every 50 iterations
        if iter_num % 50 == 0:
            val_loss = evaluate_model(model, val_data, seq_len=seq_len, model_name=model_name)
            val_losses.append(val_loss)

            print(f"[{model_name}] Iter {iter_num:5d} | Train Loss: {loss.item():.4f} | "
                  f"Val Loss: {val_loss:.4f} | LR: {current_lr:.2e} | Speed: {speed:.0f} tok/s")

        # Save checkpoint
        if iter_num > 0 and iter_num % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{iter_num}.pt"
            torch.save({
                'iteration': iter_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': loss.item(),
                'val_loss': val_losses[-1] if val_losses else None,
            }, checkpoint_path)
            print(f"[CHECKPOINT] Saved to {checkpoint_path}")

    # Final evaluation
    final_val_loss = evaluate_model(model, val_data, seq_len=seq_len, model_name=model_name)
    val_losses.append(final_val_loss)

    total_time = time.time() - start_time
    print(f"[{model_name}] Training complete in {total_time:.2f}s")

    return {
        'losses': losses,
        'val_losses': val_losses,
        'tokens_per_sec': tokens_per_sec,
        'learning_rates': learning_rates,
        'iterations': iterations,
        'total_time': total_time,
        'final_val_loss': final_val_loss
    }

def run_ultra_extended_experiment() -> Dict:
    """Run ultra-extended training experiment."""

    print("="*80)
    print("ULTRA-EXTENDED TRAINING EXPERIMENT")
    print("10,000 Iterations - Maximum Training Duration")
    print("="*80)

    # Generate ultra-large dataset
    train_data, val_data = generate_ultra_dataset(num_tokens=2000000)

    # Configure models (same as Experiment 1)
    vanilla_config = VanillaConfig(
        vocab_size=5000,
        max_seq_len=128,
        d_model=192,
        n_layers=4,
        n_heads=4,
        d_ff=768
    )

    cosmic_config = CosmicConfig(
        vocab_size=5000,
        max_seq_len=128,
        d_model=192,
        n_layers=4,
        n_heads=4
    )

    print("\n[MODEL] Creating models...")
    vanilla_model = VanillaTransformer(vanilla_config)
    cosmic_model = CosmicSynapseTransformer(cosmic_config)

    print(f"Vanilla: {vanilla_model.get_num_params():,} params")
    print(f"Cosmic: {cosmic_model.get_num_params():,} params")

    # Train both for 10,000 iterations
    vanilla_metrics = train_with_checkpoints(
        vanilla_model, train_data, val_data,
        num_iters=10000, batch_size=16, seq_len=64,
        model_name="Vanilla-Ultra"
    )

    cosmic_metrics = train_with_checkpoints(
        cosmic_model, train_data, val_data,
        num_iters=10000, batch_size=16, seq_len=64,
        model_name="12D-CST-Ultra"
    )

    results = {
        'vanilla': {
            'config': asdict(vanilla_config),
            'params': vanilla_model.get_num_params(),
            'metrics': vanilla_metrics
        },
        'cosmic': {
            'config': asdict(cosmic_config),
            'params': cosmic_model.get_num_params(),
            'metrics': cosmic_metrics
        }
    }

    return results

def plot_convergence_curves(results: Dict) -> None:
    """Generate convergence plots."""

    if not HAS_MATPLOTLIB:
        print("\n[PLOT] Skipping plots - matplotlib not available")
        return

    print("\n[PLOT] Generating convergence curves...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    vanilla_metrics = results['vanilla']['metrics']
    cosmic_metrics = results['cosmic']['metrics']

    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(vanilla_metrics['losses'], label='Vanilla', alpha=0.7)
    ax.plot(cosmic_metrics['losses'], label='12D CST', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Convergence (10K iterations)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax = axes[0, 1]
    vanilla_val_iters = np.arange(0, len(vanilla_metrics['val_losses'])) * 50
    cosmic_val_iters = np.arange(0, len(cosmic_metrics['val_losses'])) * 50
    ax.plot(vanilla_val_iters, vanilla_metrics['val_losses'], label='Vanilla', marker='o', markersize=2)
    ax.plot(cosmic_val_iters, cosmic_metrics['val_losses'], label='12D CST', marker='s', markersize=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Learning Rate Schedule
    ax = axes[1, 0]
    ax.plot(vanilla_metrics['learning_rates'], label='Learning Rate (φ-modulated)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Training Speed
    ax = axes[1, 1]
    ax.plot(vanilla_metrics['tokens_per_sec'], label='Vanilla', alpha=0.5)
    ax.plot(cosmic_metrics['tokens_per_sec'], label='12D CST', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Tokens/Second')
    ax.set_title('Training Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path("benchmark_results") / "ultra_extended_convergence.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[PLOT] Saved to {plot_path}")

def generate_ultra_report(results: Dict) -> str:
    """Generate ultra-extended training report."""

    vanilla = results['vanilla']
    cosmic = results['cosmic']

    v_final_val = vanilla['metrics']['final_val_loss']
    c_final_val = cosmic['metrics']['final_val_loss']

    improvement = ((v_final_val - c_final_val) / v_final_val) * 100

    report = f"""# 🚀 ULTRA-EXTENDED TRAINING RESULTS

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Training Duration**: 10,000 iterations (5x standard)
**Dataset**: Ultra-large synthetic (2M tokens)
**Device**: {'CUDA' if torch.cuda.is_available() else 'CPU'}

---

## 🏆 FINAL RESULTS

| Metric | Vanilla | 12D CST | Winner |
|--------|---------|---------|--------|
| **Final Val Loss** | {v_final_val:.4f} | {c_final_val:.4f} | {'🏆 12D CST' if c_final_val < v_final_val else '🏆 Vanilla'} |
| **Perplexity** | {math.exp(v_final_val):.2f} | {math.exp(c_final_val):.2f} | {'🏆 12D CST' if c_final_val < v_final_val else '🏆 Vanilla'} |
| **Improvement** | - | {improvement:+.2f}% | - |
| **Parameters** | {vanilla['params']:,} | {cosmic['params']:,} | - |
| **Training Time** | {vanilla['metrics']['total_time']:.1f}s | {cosmic['metrics']['total_time']:.1f}s | - |

---

## 📈 CONVERGENCE ANALYSIS

### Validation Loss Progression:

**Vanilla Transformer:**
- Start: {vanilla['metrics']['val_losses'][0]:.4f}
- 2000 iters: {vanilla['metrics']['val_losses'][40]:.4f}
- 5000 iters: {vanilla['metrics']['val_losses'][100]:.4f}
- 10000 iters: {v_final_val:.4f}

**12D Cosmic Synapse Transformer:**
- Start: {cosmic['metrics']['val_losses'][0]:.4f}
- 2000 iters: {cosmic['metrics']['val_losses'][40]:.4f}
- 5000 iters: {cosmic['metrics']['val_losses'][100]:.4f}
- 10000 iters: {c_final_val:.4f}

---

## 🔬 KEY INSIGHTS

1. **Long Training Benefits**: {'12D CST shows superior performance with extended training' if c_final_val < v_final_val else 'Vanilla maintains performance advantage'}
2. **Parameter Efficiency**: 12D CST achieves results with {((vanilla['params'] - cosmic['params']) / vanilla['params'] * 100):.1f}% fewer parameters
3. **Average Training Speed**:
   - Vanilla: {np.mean(vanilla['metrics']['tokens_per_sec']):.0f} tok/s
   - 12D CST: {np.mean(cosmic['metrics']['tokens_per_sec']):.0f} tok/s

---

## 💡 CONCLUSION

{'The 12D Cosmic Synapse Transformer demonstrates clear superiority with ultra-extended training, validating the hypothesis that its complex dynamics (x₁₂ states, Hebbian attention, memory module) require more iterations to fully optimize but deliver superior final performance.' if c_final_val < v_final_val else 'Results show competitive performance between architectures at extreme training durations.'}

**Training Efficiency**: {improvement:+.2f}% {'advantage' if improvement > 0 else 'difference'} for 12D CST

---

Detailed convergence plots: `ultra_extended_convergence.png`
Raw data: `ultra_extended_results.json`
Checkpoints: `ultra_checkpoints/`
"""

    return report

if __name__ == "__main__":
    print("\n🚀 Starting ULTRA-EXTENDED training experiment...")
    print("   This will take ~90-120 minutes")
    print("   Training for 10,000 iterations with checkpointing\n")

    results = run_ultra_extended_experiment()

    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    # Convert for JSON
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results_json = convert_for_json(results)

    with open(output_dir / "ultra_extended_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Generate plots
    plot_convergence_curves(results)

    # Generate report
    report = generate_ultra_report(results)

    with open(output_dir / "ULTRA_EXTENDED_REPORT.md", "w") as f:
        f.write(report)

    print("\n" + "="*80)
    print("✅ ULTRA-EXTENDED TRAINING COMPLETE!")
    print("="*80)
    print(report)
    print(f"\nResults saved to: {output_dir}/")
