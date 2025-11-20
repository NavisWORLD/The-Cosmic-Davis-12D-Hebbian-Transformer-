#!/usr/bin/env python
"""
Example 2: Train on Synthetic Data

This example demonstrates complete training on synthetic data:
1. Generate synthetic training data
2. Load configuration from YAML
3. Create and train model
4. Save checkpoint
5. Evaluate and generate text

Runtime: ~5 minutes on CPU for tiny model

Author: Cory Shane Davis
"""

import torch
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmic_synapse_transformer import CosmicConfig, CosmicSynapseTransformer
from config_loader import load_config, create_default_config
from generate_synthetic_data import SyntheticDataGenerator
import struct
import numpy as np

print("=" * 80)
print("12D COSMIC SYNAPSE TRANSFORMER - TRAINING ON SYNTHETIC DATA")
print("=" * 80)

# Configuration
USE_SAVED_DATA = False  # Set to True if you've already run generate_synthetic_data.py
DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints/example")
NUM_TOKENS = 50000  # Small dataset for quick demo

# Step 1: Generate or load synthetic data
print("\n1️⃣  Preparing synthetic data...")

if not USE_SAVED_DATA:
    print("   Generating fresh synthetic data...")
    generator = SyntheticDataGenerator(seed=42)

    # Generate tokens
    tokens = generator.generate_tokens(NUM_TOKENS, add_phi_patterns=True)
    print(f"   Generated {len(tokens):,} tokens")

    # Build vocabulary
    generator.build_vocabulary(tokens, max_vocab_size=1000)
    print(f"   Vocabulary size: {generator.vocab_size}")

    # Convert to IDs and split
    token_ids = generator.tokens_to_ids(tokens)
    split_idx = int(len(token_ids) * 0.9)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    print(f"   Train: {len(train_ids):,} tokens")
    print(f"   Val: {len(val_ids):,} tokens")
else:
    print("   Loading pre-generated data...")
    # In practice, load from .bin files created by generate_synthetic_data.py

# Step 2: Create model configuration
print("\n2️⃣  Creating model...")
config = create_default_config('tiny', 'cpu')
print(f"   Model: {config.model.d_model}D, {config.model.n_layers} layers")

model = CosmicSynapseTransformer(CosmicConfig(**config.model.__dict__))
print(f"   Parameters: {model.get_num_params():,}")

# Step 3: Setup training
print("\n3️⃣  Setting up training...")
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.training.learning_rate,
    weight_decay=config.training.weight_decay,
    betas=(config.training.beta1, config.training.beta2)
)

# Create simple dataset
def create_batches(data, seq_len, batch_size):
    """Create training batches."""
    batches = []
    for i in range(0, len(data) - seq_len, seq_len):
        if len(batches) * batch_size >= 100:  # Limit for demo
            break
        batch_data = data[i:i + seq_len]
        if len(batch_data) == seq_len:
            batches.append(torch.tensor(batch_data, dtype=torch.long))
    return batches

if not USE_SAVED_DATA:
    train_batches = create_batches(train_ids, config.model.max_seq_len, config.training.batch_size)
    val_batches = create_batches(val_ids, config.model.max_seq_len, config.training.batch_size)
    print(f"   Train batches: {len(train_batches)}")
    print(f"   Val batches: {len(val_batches)}")

# Step 4: Training loop
print("\n4️⃣  Training...")
num_epochs = 5
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    num_train_batches = min(20, len(train_batches)) if not USE_SAVED_DATA else 20

    for i in tqdm(range(num_train_batches), desc=f"Epoch {epoch + 1}/{num_epochs}"):
        if not USE_SAVED_DATA:
            batch = train_batches[i].unsqueeze(0)
        else:
            batch = torch.randint(0, 1000, (1, config.model.max_seq_len))

        optimizer.zero_grad()
        logits, loss = model(batch, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / num_train_batches

    # Validation
    model.eval()
    val_loss = 0.0
    num_val_batches = min(5, len(val_batches)) if not USE_SAVED_DATA else 5

    with torch.no_grad():
        for i in range(num_val_batches):
            if not USE_SAVED_DATA:
                batch = val_batches[i].unsqueeze(0)
            else:
                batch = torch.randint(0, 1000, (1, config.model.max_seq_len))

            logits, loss = model(batch, batch)
            val_loss += loss.item()

    avg_val_loss = val_loss / num_val_batches

    print(f"   Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, "
          f"val_loss={avg_val_loss:.4f}, x12_mean={model.x12.abs().mean():.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config.model.__dict__,
            'epoch': epoch,
            'val_loss': avg_val_loss,
        }
        torch.save(checkpoint, CHECKPOINT_DIR / 'best_model.pt')
        print(f"   ✓ Saved best model (val_loss={avg_val_loss:.4f})")

# Step 5: Generate text
print("\n5️⃣  Generating text...")
model.eval()

prompt = torch.randint(0, config.model.vocab_size, (1, 10))
print(f"   Prompt: {prompt.tolist()[0]}")

with torch.no_grad():
    output = model.generate(prompt, max_new_tokens=30, temperature=0.8)

print(f"   Generated: {output.tolist()[0]}")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
print(f"\nCheckpoint saved to: {CHECKPOINT_DIR / 'best_model.pt'}")
print(f"Best validation loss: {best_val_loss:.4f}")
print("\nNext steps:")
print("  - Load checkpoint with torch.load()")
print("  - Run inference_cosmic_transformer.py for deployment")
print("  - See examples/03_interactive_demo.py for visualization")
