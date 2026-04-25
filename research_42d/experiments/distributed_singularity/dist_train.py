"""
DISTRIBUTED SINGULARITY TRAINER (DDP)
=====================================
Production-ready Distributed Data Parallel (DDP) training script for 
Cosmic Synapse (12D) and Hyper-Cosmic (42D) models.

Usage:
    torchrun --nproc_per_node=NUM_GPUS dist_train.py --model 42D --batch_size 32

Features:
- Multi-GPU / Multi-Node support via PyTorch DDP
- DistributedSampler for efficient data sharding
- Synchronized Batch Norm (if applicable)
- Rank-0 logging and checkpointing
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import argparse
from pathlib import Path
import time

# Add package paths
ROOT_DIR = Path(__file__).parents[3]
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
# Import 42D model from research folder (need to add path)
sys.path.append(str(ROOT_DIR / "research_42d" / "experiments" / "singularity_42d"))
from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig

def setup():
    """Initialize the distributed process group."""
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

def cleanup():
    """Destroy the process group."""
    dist.destroy_process_group()

class SyntheticDataset(Dataset):
    """Simple wrapper for synthetic data generator compatible with DataLoader."""
    def __init__(self, generator, vocab_size, seq_len, num_samples=10000):
        self.generator = generator
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate on the fly (CPU) -> Send to GPU in loop
        # In a real scenario, this would load from disk
        x, y = self.generator.generate_batch(1, self.seq_len, self.vocab_size)
        return x.squeeze(0), y.squeeze(0)

def train(rank, args):
    # Setup DDP
    setup()
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        
    # 1. Initialize Tokenizer/Vocab
    # In production, load from file. Here we simulate or load.
    gen = SyntheticDataGenerator()
    # For demo, small vocab. In prod, load 30k vocab.
    gen.build_vocabulary(["hello", "world", "science", "logic", "chaos", "entropy"]) 
    vocab_size = gen.vocab_size

    # 2. Initialize Model
    if args.model == "42D":
        config = HyperConfig(vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads)
        model = HyperCosmicTransformer(config)
    else:
        config = CosmicConfig(vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads)
        model = CosmicSynapseTransformer(config)

    model.to(device)
    
    # Wrap in DDP
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    # 3. Data Loader with DistributedSampler
    dataset = SyntheticDataset(gen, vocab_size, args.seq_len)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 5. Training Loop
    if rank == 0:
        print(f"🚀 Starting Distributed Training: {args.model} on {dist.get_world_size()} devices")
        print(f"   Batch Size: {args.batch_size} (Total: {args.batch_size * dist.get_world_size()})")

    start_time = time.time()
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch) # Crucial for shuffling
        
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            if args.model == "42D":
                logits, loss, _ = model(x, targets=y)
            else:
                logits, loss, _ = model(x, targets=y)
                
            # Backward
            loss.backward()
            optimizer.step()
            
            if rank == 0 and step % 10 == 0:
                print(f"[Epoch {epoch} | Step {step}] Loss: {loss.item():.4f}")

    if rank == 0:
        print(f"✅ Training Complete. Time: {time.time() - start_time:.2f}s")
        # Save checkpoint (only on rank 0)
        save_path = Path(f"checkpoint_{args.model.lower()}_ddp.pt")
        torch.save(model.module.state_dict(), save_path)
        print(f"💾 Model saved to {save_path}")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="42D", choices=["12D", "42D"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    
    # Torchrun handles rank/world_size automatically via env vars
    # We just need to parse normal args
    
    args = parser.parse_args()
    
    # Get rank from env (set by torchrun)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    
    train(rank, args)
