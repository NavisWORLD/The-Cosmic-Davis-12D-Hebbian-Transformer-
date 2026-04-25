"""
COSMIC DAVIS vs VANILLA - REAL DATASET BENCHMARK
==================================================
Using WikiText-2, the standard academic benchmark for language models.

This is the kind of evidence that gets published in papers.
"""

import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass

# Setup paths
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig

# Try to import datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Installing datasets library...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "-q"])
    from datasets import load_dataset

from transformers import GPT2Tokenizer

# ============================================================
# VANILLA TRANSFORMER (Same as before)
# ============================================================

@dataclass
class VanillaConfig:
    vocab_size: int = 50257  # GPT-2 vocab size
    max_seq_len: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1


class VanillaAttention(nn.Module):
    def __init__(self, config):
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
    def __init__(self, config):
        super().__init__()
        self.W1 = nn.Linear(config.d_model, config.d_ff)
        self.W2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.W2(self.dropout(F.gelu(self.W1(x))))


class VanillaLayer(nn.Module):
    def __init__(self, config):
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
    def __init__(self, config):
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
        print(f"[VANILLA] {n_params/1e6:.2f}M parameters")
    
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


# ============================================================
# BENCHMARK ON WIKITEXT-2
# ============================================================

def run_wikitext_benchmark():
    print("="*70)
    print("🔬 COSMIC DAVIS vs VANILLA - WIKITEXT-2 BENCHMARK")
    print("="*70)
    
    # Load WikiText-2
    print("\n📚 Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    print(f"   Train: {len(dataset['train'])} samples")
    print(f"   Valid: {len(dataset['validation'])} samples")
    print(f"   Test:  {len(dataset['test'])} samples")
    
    # Load tokenizer
    print("\n🔤 Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    vocab_size = tokenizer.vocab_size
    print(f"   Vocab size: {vocab_size}")
    
    # Configuration
    d_model = 256
    n_layers = 6
    n_heads = 8
    seq_len = 128
    batch_size = 8
    n_epochs = 1
    
    print(f"\n📊 Configuration:")
    print(f"   d_model: {d_model}")
    print(f"   n_layers: {n_layers}")
    print(f"   n_heads: {n_heads}")
    print(f"   seq_len: {seq_len}")
    print(f"   batch_size: {batch_size}")
    
    # Tokenize dataset
    print("\n🔧 Tokenizing dataset...")
    
    # Combine all text
    train_text = " ".join([x["text"] for x in dataset["train"] if x["text"].strip()])
    valid_text = " ".join([x["text"] for x in dataset["validation"] if x["text"].strip()])
    
    train_tokens = tokenizer.encode(train_text)[:50000]  # Limit for speed
    valid_tokens = tokenizer.encode(valid_text)[:10000]
    
    print(f"   Train tokens: {len(train_tokens)}")
    print(f"   Valid tokens: {len(valid_tokens)}")
    
    # Create batches
    def create_batches(tokens, seq_len, batch_size):
        batches = []
        for i in range(0, len(tokens) - seq_len - 1, seq_len):
            x = tokens[i:i+seq_len]
            y = tokens[i+1:i+seq_len+1]
            if len(x) == seq_len and len(y) == seq_len:
                batches.append((x, y))
        
        # Group into batches
        final_batches = []
        for i in range(0, len(batches) - batch_size, batch_size):
            batch = batches[i:i+batch_size]
            x = torch.tensor([b[0] for b in batch], dtype=torch.long)
            y = torch.tensor([b[1] for b in batch], dtype=torch.long)
            final_batches.append((x, y))
        
        return final_batches
    
    train_batches = create_batches(train_tokens, seq_len, batch_size)
    valid_batches = create_batches(valid_tokens, seq_len, batch_size)
    
    print(f"   Train batches: {len(train_batches)}")
    print(f"   Valid batches: {len(valid_batches)}")
    
    # Create models
    print("\n🔧 Creating Models...")
    
    vanilla_config = VanillaConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 4
    )
    vanilla_model = VanillaTransformer(vanilla_config)
    
    cosmic_config = CosmicConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads
    )
    cosmic_config.d_model = d_model
    cosmic_config.d_ff = d_model * 4
    cosmic_config.d_k = d_model // n_heads
    cosmic_model = CosmicSynapseTransformer(cosmic_config)
    
    # Count parameters
    vanilla_params = sum(p.numel() for p in vanilla_model.parameters())
    cosmic_params = sum(p.numel() for p in cosmic_model.parameters())
    
    print(f"\n📈 Parameter Count:")
    print(f"   Vanilla: {vanilla_params:,} ({vanilla_params/1e6:.2f}M)")
    print(f"   Cosmic:  {cosmic_params:,} ({cosmic_params/1e6:.2f}M)")
    
    # Optimizers
    vanilla_opt = torch.optim.AdamW(vanilla_model.parameters(), lr=3e-4)
    cosmic_opt = torch.optim.AdamW(cosmic_model.parameters(), lr=3e-4)
    
    # Training
    print("\n🏋️ Training on WikiText-2...")
    print("-" * 70)
    
    vanilla_train_losses = []
    cosmic_train_losses = []
    
    for epoch in range(n_epochs):
        vanilla_model.train()
        cosmic_model.train()
        
        for i, (x, y) in enumerate(train_batches):
            # Train Vanilla
            vanilla_opt.zero_grad()
            _, v_loss, _ = vanilla_model(x, targets=y)
            v_loss.backward()
            vanilla_opt.step()
            vanilla_train_losses.append(v_loss.item())
            
            # Train Cosmic
            cosmic_opt.zero_grad()
            _, c_loss, _ = cosmic_model(x, targets=y)
            c_loss.backward()
            cosmic_opt.step()
            cosmic_train_losses.append(c_loss.item())
            
            if (i + 1) % 20 == 0:
                v_avg = sum(vanilla_train_losses[-20:]) / 20
                c_avg = sum(cosmic_train_losses[-20:]) / 20
                print(f"   Batch {i+1:3d}/{len(train_batches)}: Vanilla={v_avg:.4f} | Cosmic={c_avg:.4f} | Δ={c_avg-v_avg:+.4f}")
    
    # Validation
    print("\n📊 Validating on held-out data...")
    
    vanilla_model.eval()
    cosmic_model.eval()
    
    vanilla_val_losses = []
    cosmic_val_losses = []
    
    with torch.no_grad():
        for x, y in valid_batches:
            _, v_loss, _ = vanilla_model(x, targets=y)
            _, c_loss, _ = cosmic_model(x, targets=y)
            vanilla_val_losses.append(v_loss.item())
            cosmic_val_losses.append(c_loss.item())
    
    v_val = sum(vanilla_val_losses) / len(vanilla_val_losses)
    c_val = sum(cosmic_val_losses) / len(cosmic_val_losses)
    
    v_ppl = math.exp(min(v_val, 10))
    c_ppl = math.exp(min(c_val, 10))
    
    # Results
    print("\n" + "="*70)
    print("📊 WIKITEXT-2 BENCHMARK RESULTS")
    print("="*70)
    
    print(f"\n🎯 Validation Loss:")
    print(f"   Vanilla:  {v_val:.4f}")
    print(f"   Cosmic:   {c_val:.4f}")
    
    print(f"\n📈 Perplexity (lower is better):")
    print(f"   Vanilla:  {v_ppl:.2f}")
    print(f"   Cosmic:   {c_ppl:.2f}")
    
    if c_val < v_val:
        improvement = (1 - c_val/v_val) * 100
        ppl_improvement = (1 - c_ppl/v_ppl) * 100
        print(f"\n🏆 COSMIC DAVIS WINS!")
        print(f"   Loss improvement: {improvement:.2f}%")
        print(f"   Perplexity improvement: {ppl_improvement:.2f}%")
    else:
        print(f"\n   Vanilla wins on this benchmark")
    
    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)
    
    return {
        "vanilla_val_loss": v_val,
        "cosmic_val_loss": c_val,
        "vanilla_ppl": v_ppl,
        "cosmic_ppl": c_ppl
    }


if __name__ == "__main__":
    results = run_wikitext_benchmark()
