"""
COMPREHENSIVE REAL-WORLD-SCALE BENCHMARK
========================================

Full benchmark suite comparing 12D CST vs Vanilla Transformer:
1. Large-scale data (1M tokens - 20x previous test)
2. Extended training (2000 iterations - 4x previous test)
3. Parameter-matched comparison
4. Multiple model sizes
5. Convergence analysis

This provides production-ready insights.
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

from cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from benchmark_transformer import (
    VanillaTransformer, VanillaConfig,
    train_model, evaluate_model
)

def generate_large_dataset(num_tokens: int = 1000000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate large-scale realistic synthetic data."""
    print(f"\n[DATA] Generating {num_tokens:,} tokens (large-scale)...")

    from generate_synthetic_data import SyntheticDataGenerator

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

def run_comprehensive_benchmark() -> Dict:
    """Run comprehensive benchmark suite."""

    print("="*80)
    print("COMPREHENSIVE REAL-WORLD-SCALE BENCHMARK")
    print("12D Cosmic Synapse Transformer vs Vanilla Transformer")
    print("="*80)

    # Generate large dataset
    train_data, val_data = generate_large_dataset(num_tokens=1000000)

    results = {}

    # ===================================================================
    # EXPERIMENT 1: Extended Training (2000 iterations)
    # ===================================================================

    print("\n" + "="*80)
    print("EXPERIMENT 1: Extended Training (2000 iterations)")
    print("="*80)

    vanilla_config_ext = VanillaConfig(
        vocab_size=5000,
        max_seq_len=128,
        d_model=192,
        n_layers=4,
        n_heads=4,
        d_ff=768
    )

    cosmic_config_ext = CosmicConfig(
        vocab_size=5000,
        max_seq_len=128,
        d_model=192,
        n_layers=4,
        n_heads=4
    )

    print("\n[MODEL] Creating models for extended training...")
    vanilla_ext = VanillaTransformer(vanilla_config_ext)
    cosmic_ext = CosmicSynapseTransformer(cosmic_config_ext)

    print(f"Vanilla: {vanilla_ext.get_num_params():,} params")
    print(f"Cosmic: {cosmic_ext.get_num_params():,} params")

    # Train both for 2000 iterations
    vanilla_metrics_ext = train_model(
        vanilla_ext, train_data, val_data,
        num_iters=2000, batch_size=16, seq_len=64,
        model_name="Vanilla-Extended"
    )

    cosmic_metrics_ext = train_model(
        cosmic_ext, train_data, val_data,
        num_iters=2000, batch_size=16, seq_len=64,
        model_name="12D CST-Extended"
    )

    results['extended_training'] = {
        'vanilla': {
            'config': asdict(vanilla_config_ext),
            'params': vanilla_ext.get_num_params(),
            'metrics': vanilla_metrics_ext
        },
        'cosmic': {
            'config': asdict(cosmic_config_ext),
            'params': cosmic_ext.get_num_params(),
            'metrics': cosmic_metrics_ext
        }
    }

    # ===================================================================
    # EXPERIMENT 2: Parameter-Matched Comparison
    # ===================================================================

    print("\n" + "="*80)
    print("EXPERIMENT 2: Parameter-Matched Comparison")
    print("="*80)

    # Increase 12D CST d_model to match vanilla's params
    # Vanilla has ~2M params, need to find d_model for cosmic to match

    # Try d_model = 256
    cosmic_config_matched = CosmicConfig(
        vocab_size=5000,
        max_seq_len=128,
        d_model=256,  # Increased from 192
        n_layers=4,
        n_heads=4
    )

    print("\n[MODEL] Creating parameter-matched models...")
    vanilla_matched = VanillaTransformer(vanilla_config_ext)  # Same as before
    cosmic_matched = CosmicSynapseTransformer(cosmic_config_matched)

    v_params = vanilla_matched.get_num_params()
    c_params = cosmic_matched.get_num_params()

    print(f"Vanilla: {v_params:,} params")
    print(f"Cosmic (matched): {c_params:,} params")
    print(f"Difference: {abs(v_params - c_params):,} ({abs(v_params - c_params)/v_params*100:.1f}%)")

    # Train both
    vanilla_metrics_matched = train_model(
        vanilla_matched, train_data, val_data,
        num_iters=2000, batch_size=16, seq_len=64,
        model_name="Vanilla-Matched"
    )

    cosmic_metrics_matched = train_model(
        cosmic_matched, train_data, val_data,
        num_iters=2000, batch_size=16, seq_len=64,
        model_name="12D CST-Matched"
    )

    results['parameter_matched'] = {
        'vanilla': {
            'config': asdict(vanilla_config_ext),
            'params': v_params,
            'metrics': vanilla_metrics_matched
        },
        'cosmic': {
            'config': asdict(cosmic_config_matched),
            'params': c_params,
            'metrics': cosmic_metrics_matched
        }
    }

    # ===================================================================
    # EXPERIMENT 3: Scaling Analysis (Different Sizes)
    # ===================================================================

    print("\n" + "="*80)
    print("EXPERIMENT 3: Scaling Analysis (Tiny, Small, Medium)")
    print("="*80)

    sizes = [
        ('tiny', {'d_model': 128, 'n_layers': 2, 'n_heads': 2}),
        ('small', {'d_model': 192, 'n_layers': 4, 'n_heads': 4}),
        ('medium', {'d_model': 256, 'n_layers': 6, 'n_heads': 4})
    ]

    results['scaling'] = {}

    for size_name, size_config in sizes:
        print(f"\n[SCALING] Testing {size_name} size...")

        v_config = VanillaConfig(
            vocab_size=5000, max_seq_len=128,
            d_model=size_config['d_model'],
            n_layers=size_config['n_layers'],
            n_heads=size_config['n_heads'],
            d_ff=size_config['d_model'] * 4
        )

        c_config = CosmicConfig(
            vocab_size=5000, max_seq_len=128,
            d_model=size_config['d_model'],
            n_layers=size_config['n_layers'],
            n_heads=size_config['n_heads']
        )

        v_model = VanillaTransformer(v_config)
        c_model = CosmicSynapseTransformer(c_config)

        print(f"  Vanilla: {v_model.get_num_params():,} params")
        print(f"  Cosmic: {c_model.get_num_params():,} params")

        # Shorter training for scaling test (1000 iters)
        v_metrics = train_model(
            v_model, train_data, val_data,
            num_iters=1000, batch_size=16, seq_len=64,
            model_name=f"Vanilla-{size_name}"
        )

        c_metrics = train_model(
            c_model, train_data, val_data,
            num_iters=1000, batch_size=16, seq_len=64,
            model_name=f"12D CST-{size_name}"
        )

        results['scaling'][size_name] = {
            'vanilla': {
                'config': asdict(v_config),
                'params': v_model.get_num_params(),
                'metrics': v_metrics
            },
            'cosmic': {
                'config': asdict(c_config),
                'params': c_model.get_num_params(),
                'metrics': c_metrics
            }
        }

    return results

def generate_comprehensive_report(results: Dict) -> str:
    """Generate comprehensive markdown report."""

    report = f"""# 🔬 COMPREHENSIVE REAL-WORLD-SCALE BENCHMARK RESULTS

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Dataset**: Large-scale synthetic (1M tokens)
**Device**: {'CUDA' if torch.cuda.is_available() else 'CPU'}
**PyTorch**: {torch.__version__}

---

## 📊 EXPERIMENT 1: Extended Training (2000 iterations)

**Purpose**: Test convergence over longer training runs

"""

    # Extended training results
    ext = results['extended_training']
    v_final_loss = ext['vanilla']['metrics']['losses'][-1]
    c_final_loss = ext['cosmic']['metrics']['losses'][-1]
    v_final_val = ext['vanilla']['metrics']['val_losses'][-1]
    c_final_val = ext['cosmic']['metrics']['val_losses'][-1]

    report += f"""
| Metric | Vanilla | 12D CST | Winner |
|--------|---------|---------|--------|
| **Parameters** | {ext['vanilla']['params']:,} | {ext['cosmic']['params']:,} | - |
| **Final Train Loss** | {v_final_loss:.4f} | {c_final_loss:.4f} | {'Vanilla' if v_final_loss < c_final_loss else '12D CST'} |
| **Final Val Loss** | {v_final_val:.4f} | {c_final_val:.4f} | {'Vanilla' if v_final_val < c_final_val else '12D CST'} |
| **Perplexity** | {math.exp(v_final_val):.2f} | {math.exp(c_final_val):.2f} | {'Vanilla' if v_final_val < c_final_val else '12D CST'} |
| **Avg Speed** | {np.mean(ext['vanilla']['metrics']['tokens_per_sec']):.0f} tok/s | {np.mean(ext['cosmic']['metrics']['tokens_per_sec']):.0f} tok/s | - |

**Convergence**:
- Vanilla: {ext['vanilla']['metrics']['losses'][0]:.4f} → {v_final_loss:.4f} (Δ {ext['vanilla']['metrics']['losses'][0] - v_final_loss:.4f})
- 12D CST: {ext['cosmic']['metrics']['losses'][0]:.4f} → {c_final_loss:.4f} (Δ {ext['cosmic']['metrics']['losses'][0] - c_final_loss:.4f})

---

## 📊 EXPERIMENT 2: Parameter-Matched Comparison

**Purpose**: Fair architectural comparison with equal parameters

"""

    # Parameter-matched results
    matched = results['parameter_matched']
    v_params = matched['vanilla']['params']
    c_params = matched['cosmic']['params']
    v_final_loss_m = matched['vanilla']['metrics']['losses'][-1]
    c_final_loss_m = matched['cosmic']['metrics']['losses'][-1]
    v_final_val_m = matched['vanilla']['metrics']['val_losses'][-1]
    c_final_val_m = matched['cosmic']['metrics']['val_losses'][-1]

    report += f"""
| Metric | Vanilla | 12D CST | Winner |
|--------|---------|---------|--------|
| **Parameters** | {v_params:,} | {c_params:,} | {'Matched' if abs(v_params - c_params) < v_params * 0.05 else 'Unmatched'} |
| **Final Train Loss** | {v_final_loss_m:.4f} | {c_final_loss_m:.4f} | {'🏆 Vanilla' if v_final_loss_m < c_final_loss_m else '🏆 12D CST'} |
| **Final Val Loss** | {v_final_val_m:.4f} | {c_final_val_m:.4f} | {'🏆 Vanilla' if v_final_val_m < c_final_val_m else '🏆 12D CST'} |
| **Perplexity** | {math.exp(v_final_val_m):.2f} | {math.exp(c_final_val_m):.2f} | {'🏆 Vanilla' if v_final_val_m < c_final_val_m else '🏆 12D CST'} |

---

## 📊 EXPERIMENT 3: Scaling Analysis

**Purpose**: Test how architectures scale across model sizes

"""

    for size_name in ['tiny', 'small', 'medium']:
        if size_name in results['scaling']:
            scaling = results['scaling'][size_name]
            v_loss = scaling['vanilla']['metrics']['losses'][-1]
            c_loss = scaling['cosmic']['metrics']['losses'][-1]
            v_val = scaling['vanilla']['metrics']['val_losses'][-1]
            c_val = scaling['cosmic']['metrics']['val_losses'][-1]

            report += f"""
### {size_name.upper()} Models

| Metric | Vanilla | 12D CST | Winner |
|--------|---------|---------|--------|
| Parameters | {scaling['vanilla']['params']:,} | {scaling['cosmic']['params']:,} | - |
| Train Loss | {v_loss:.4f} | {c_loss:.4f} | {'Vanilla' if v_loss < c_loss else '12D CST'} |
| Val Loss | {v_val:.4f} | {c_val:.4f} | {'Vanilla' if v_val < c_val else '12D CST'} |
| Perplexity | {math.exp(v_val):.2f} | {math.exp(c_val):.2f} | {'Vanilla' if v_val < c_val else '12D CST'} |

"""

    report += """
---

## 🏆 OVERALL CONCLUSIONS

[Analysis will be provided in detailed discussion]

---

Raw data saved to: `comprehensive_benchmark_results.json`
"""

    return report

if __name__ == "__main__":
    # Run comprehensive benchmark
    print("\n🚀 Starting comprehensive real-world-scale benchmark...")
    print("   This will take significant time (~30-60 minutes)")
    print("   Running 3 experiments with multiple model sizes\n")

    results = run_comprehensive_benchmark()

    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    # Convert numpy arrays to lists for JSON
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

    with open(output_dir / "comprehensive_benchmark_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Generate report
    report = generate_comprehensive_report(results)

    with open(output_dir / "COMPREHENSIVE_BENCHMARK_REPORT.md", "w") as f:
        f.write(report)

    print("\n" + "="*80)
    print("✅ COMPREHENSIVE BENCHMARK COMPLETE!")
    print("="*80)
    print(report)
    print(f"\nResults saved to: {output_dir}/")
