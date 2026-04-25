"""
DOWNLOAD PRE-TRAINED WEIGHTS FOR COSMIC DAVIS
==============================================
Downloads GPT-2 weights from HuggingFace and saves them locally.
These will be used to supercharge the Cosmic Davis model.
"""

import os
import sys
from pathlib import Path

print("="*60)
print("🚀 COSMIC DAVIS - DOWNLOADING PRE-TRAINED WEIGHTS")
print("="*60)

# Check transformers
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
    print("✅ Transformers library found")
except ImportError:
    print("❌ Installing transformers...")
    os.system("pip install transformers")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

import torch

# Create weights directory
weights_dir = Path("pretrained_weights")
weights_dir.mkdir(exist_ok=True)

# Available models
MODELS = {
    "gpt2": {"params": "117M", "size": "~500MB", "recommended": True},
    "gpt2-medium": {"params": "345M", "size": "~1.5GB", "recommended": False},
    "gpt2-large": {"params": "774M", "size": "~3GB", "recommended": False},
    "distilgpt2": {"params": "82M", "size": "~350MB", "recommended": False},
}

print("\n📦 Available Models:")
for name, info in MODELS.items():
    rec = " ⭐ RECOMMENDED" if info["recommended"] else ""
    print(f"   - {name}: {info['params']} params, {info['size']}{rec}")

# Download the recommended model
model_name = "gpt2"  # 117M params - good for CPU
print(f"\n📥 Downloading {model_name}...")
print("   This may take a few minutes on first run...")

# Download model and tokenizer
print("\n[1/3] Downloading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(weights_dir / model_name)
print(f"   ✅ Tokenizer saved to {weights_dir / model_name}")

print("\n[2/3] Downloading model weights...")
model = GPT2LMHeadModel.from_pretrained(model_name)
model.save_pretrained(weights_dir / model_name)
print(f"   ✅ Model saved to {weights_dir / model_name}")

print("\n[3/3] Extracting architecture info...")
config = model.config
print(f"   • vocab_size: {config.vocab_size}")
print(f"   • n_positions (max_seq_len): {config.n_positions}")
print(f"   • n_embd (d_model): {config.n_embd}")
print(f"   • n_layer: {config.n_layer}")
print(f"   • n_head: {config.n_head}")

# Save config info for Cosmic Davis
config_info = {
    "model_name": model_name,
    "vocab_size": config.vocab_size,
    "max_seq_len": config.n_positions,
    "d_model": config.n_embd,
    "n_layers": config.n_layer,
    "n_heads": config.n_head,
    "d_ff": config.n_embd * 4,  # GPT-2 uses 4x expansion
}

import json
with open(weights_dir / "model_config.json", "w") as f:
    json.dump(config_info, f, indent=2)

print(f"\n   ✅ Config saved to {weights_dir / 'model_config.json'}")

# Test the model
print("\n🧪 Quick test of downloaded model...")
test_input = tokenizer("Hello, I am", return_tensors="pt")
with torch.no_grad():
    output = model.generate(**test_input, max_new_tokens=10, do_sample=True, temperature=0.7)
test_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"   Input: 'Hello, I am'")
print(f"   Output: '{test_output}'")

# Calculate total size
total_size = sum(f.stat().st_size for f in (weights_dir / model_name).rglob("*") if f.is_file())
print(f"\n📊 Total download size: {total_size / 1024 / 1024:.1f} MB")

print("\n" + "="*60)
print("✅ DOWNLOAD COMPLETE!")
print("="*60)
print(f"\nWeights saved to: {weights_dir.absolute() / model_name}")
print("\nNext step: Run the weight loader to integrate with Cosmic Davis")
print("="*60)
