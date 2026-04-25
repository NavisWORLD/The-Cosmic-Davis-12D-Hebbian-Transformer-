"""
42D SINGULARITY BENCHMARK
=========================

Tests the Hyper-Cosmic Synapse Transformer (42D) against baselines.
Focus: Can 42D internal states compensate for fewer parameters?
"""

import torch
import torch.nn as nn
import time
import math
import json
import sys
import numpy as np
from pathlib import Path
from dataclasses import asdict

# Add package to path
sys.path.append(str(Path(__file__).parents[2] / "packages" / "cosmic-synapse-transformer"))

from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig

# Import Vanilla for baseline
from benchmarks.scripts.benchmark_transformer import VanillaTransformer, VanillaConfig, train_model

def run_singularity_test():
    print("="*80)
    print("42D SINGULARITY EXPERIMENT")
    print("Hypothesis: 42D internal states > Parameters")
    print("="*80)
    
    # 1. Generate Data
    print("\n[DATA] Generating synthetic corpus...")
    gen = SyntheticDataGenerator(seed=42)
    tokens = gen.generate_tokens(100000, add_phi_patterns=True) # 100k tokens for quick test
    gen.build_vocabulary(tokens)
    token_ids = np.array(gen.tokens_to_ids(tokens), dtype=np.uint16)
    
    split = int(0.9 * len(token_ids))
    train_data = token_ids[:split]
    val_data = token_ids[split:]
    
    vocab_size = gen.vocab_size
    print(f"Vocab Size: {vocab_size}")
    
    # 2. Define Models
    
    # A. TINY 42D Model (The Challenger)
    # We make it very small in parameters, relying on 42D state
    hyper_config = HyperConfig(
        vocab_size=vocab_size,
        max_seq_len=128,
        d_model=128,      # Very small!
        n_layers=4,
        n_heads=4,
        k=0.1, gamma=0.05
    )
    hyper_model = HyperCosmicTransformer(hyper_config)
    
    # B. LARGE Vanilla Model (The Goliath)
    # Much larger parameters
    vanilla_config = VanillaConfig(
        vocab_size=vocab_size,
        max_seq_len=128,
        d_model=256,      # 2x larger dimension
        n_layers=6,       # More layers
        n_heads=4,
        d_ff=1024         # 4x expansion
    )
    vanilla_model = VanillaTransformer(vanilla_config)
    
    h_params = hyper_model.get_num_params()
    v_params = vanilla_model.get_num_params()
    
    print(f"\n[CONTENDERS]")
    print(f"1. 42D Hyper-Cosmic (Tiny): {h_params:,} params")
    print(f"2. Vanilla Transformer (Large): {v_params:,} params")
    print(f"Ratio: Vanilla has {v_params/h_params:.1f}x more parameters!")
    
    # 3. Train
    print("\n[TRAINING] 42D Hyper-Cosmic...")
    h_metrics = train_model(
        hyper_model, train_data, val_data,
        num_iters=1000, batch_size=32, seq_len=64,
        model_name="Hyper-Cosmic-42D-Tiny"
    )
    
    print("\n[TRAINING] Vanilla Goliath...")
    v_metrics = train_model(
        vanilla_model, train_data, val_data,
        num_iters=1000, batch_size=32, seq_len=64,
        model_name="Vanilla-Large"
    )
    
    # 4. Results
    h_loss = h_metrics['val_losses'][-1]
    v_loss = v_metrics['val_losses'][-1]
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"42D Tiny Loss:   {h_loss:.4f}")
    print(f"Vanilla Large Loss: {v_loss:.4f}")
    
    if h_loss < v_loss:
        print(f"\n🏆 42D WINS! (Beating a model {v_params/h_params:.1f}x its size)")
    else:
        print(f"\nVanilla Wins (Parameters still matter)")
        
    # Save
    results = {
        'hyper': {'params': h_params, 'loss': h_loss, 'metrics': h_metrics},
        'vanilla': {'params': v_params, 'loss': v_loss, 'metrics': v_metrics}
    }
    
    with open("singularity_results.json", "w") as f:
        # Helper to convert numpy/arrays
        def default(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
        json.dump(results, f, default=default, indent=2)

if __name__ == "__main__":
    run_singularity_test()
