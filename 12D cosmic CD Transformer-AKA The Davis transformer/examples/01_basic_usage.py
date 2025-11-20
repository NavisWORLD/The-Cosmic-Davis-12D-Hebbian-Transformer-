#!/usr/bin/env python
"""
Example 1: Basic Usage of 12D Cosmic Synapse Transformer

This example shows the minimal code needed to:
1. Create a tiny model
2. Generate synthetic data
3. Train for 10 steps
4. Generate text

Total runtime: ~30 seconds on CPU

Author: Cory Shane Davis
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmic_synapse_transformer import CosmicConfig, CosmicSynapseTransformer

print("=" * 80)
print("12D COSMIC SYNAPSE TRANSFORMER - BASIC USAGE EXAMPLE")
print("=" * 80)

# Step 1: Create a tiny model
print("\n1️⃣  Creating tiny model...")
config = CosmicConfig(
    vocab_size=100,
    max_seq_len=64,
    d_model=96,
    n_layers=2,
    n_heads=2,
)

model = CosmicSynapseTransformer(config)
print(f"   Model created with {model.get_num_params():,} parameters")

# Step 2: Create simple synthetic data
print("\n2️⃣  Creating synthetic training data...")
batch_size = 2
seq_len = 32
num_steps = 10

# Random token sequences (in practice, use generate_synthetic_data.py)
train_data = [torch.randint(0, config.vocab_size, (seq_len,)) for _ in range(20)]

# Step 3: Train for 10 steps
print("\n3️⃣  Training for 10 steps...")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

for step in range(num_steps):
    # Sample random batch
    batch_inputs = torch.stack([train_data[i] for i in range(batch_size)])
    batch_targets = batch_inputs.clone()

    # Training step
    optimizer.zero_grad()
    logits, loss = model(batch_inputs, batch_targets)
    loss.backward()
    optimizer.step()

    # Print progress
    print(f"   Step {step + 1}/{num_steps}: loss = {loss.item():.4f}, "
          f"x12 = {model.x12.abs().mean().item():.4f}")

print("\n4️⃣  Generating text...")
model.eval()

# Start with a simple prompt
prompt = torch.randint(0, config.vocab_size, (1, 5))
print(f"   Prompt tokens: {prompt.tolist()[0]}")

# Generate 20 new tokens
with torch.no_grad():
    output = model.generate(prompt, max_new_tokens=20, temperature=0.8)

print(f"   Generated tokens: {output.tolist()[0]}")

print("\n" + "=" * 80)
print("✅ BASIC EXAMPLE COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print("  - Run examples/02_train_on_synthetic.py for full training")
print("  - Run generate_synthetic_data.py to create real datasets")
print("  - See docs/QUICKSTART.md for more information")
