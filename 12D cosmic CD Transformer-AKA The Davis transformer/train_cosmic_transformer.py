"""
COMPLETE TRAINING PIPELINE FOR 12D COSMIC SYNAPSE TRANSFORMER
===============================================================

Full production training system with:
- Distributed training (multi-GPU)
- Efficient data loading
- Checkpointing & resumption
- Wandb/TensorBoard logging
- Evaluation metrics
- Learning rate scheduling

Author: Cory Shane Davis
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp

import os
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Any
import numpy as np

from cosmic_synapse_transformer import (
    CosmicSynapseTransformer,
    CosmicConfig,
    PHI,
    PHI_INV
)

# Optional imports for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("[WARNING] wandb not installed. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# ===================================================================
# DATASET
# ===================================================================

class TextDataset(Dataset):
    """
    Efficient text dataset for language modeling.
    Loads pre-tokenized data.
    """
    
    def __init__(self, data_path: str, block_size: int) -> None:
        """
        Args:
            data_path: Path to .bin file containing tokenized data
            block_size: Sequence length
        """
        self.block_size = block_size

        # Load data
        if data_path.endswith('.bin'):
            # Memory-mapped numpy array for efficiency
            self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        elif data_path.endswith('.npy'):
            self.data = np.load(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        print(f"[DATASET] Loaded {len(self.data):,} tokens from {data_path}")

    def __len__(self) -> int:
        # Number of complete blocks we can form
        return len(self.data) // self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get block
        start = idx * self.block_size
        end = start + self.block_size + 1

        chunk = self.data[start:end]

        # Input and target (shifted by 1)
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y

# ===================================================================
# TRAINING CONFIGURATION
# ===================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    
    # Data
    train_data_path: str = "train.bin"
    val_data_path: str = "val.bin"
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_iters: int = 100000
    
    # Optimization
    learning_rate: float = 3e-4 * PHI_INV  # φ-scaled
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    
    # Learning rate schedule
    warmup_iters: int = 2000
    lr_decay_iters: int = 100000
    min_lr: float = 3e-5 * PHI_INV
    
    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 200
    
    # Logging
    log_interval: int = 10
    
    # Checkpointing
    ckpt_dir: str = "checkpoints"
    save_interval: int = 1000
    
    # System
    device: str = "cuda"
    compile: bool = True  # torch.compile (PyTorch 2.0+)
    dtype: str = "bfloat16"  # float32, bfloat16, float16
    
    # Distributed
    backend: str = "nccl"
    
    # Wandb
    wandb_project: str = "12d-cst-transformer"
    wandb_run_name: Optional[str] = None

# ===================================================================
# LEARNING RATE SCHEDULER
# ===================================================================

def get_lr(it: int, config: 'TrainingConfig') -> float:
    """
    Cosine learning rate schedule with warmup.
    Following Chinchilla scaling laws with φ-optimization.
    """
    # Warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters

    # Decay
    if it > config.lr_decay_iters:
        return config.min_lr

    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    # φ-modulated decay for resonance
    phi_factor = 1.0 + 0.1 * math.sin(2 * math.pi * decay_ratio * PHI)

    return config.min_lr + coeff * (config.learning_rate - config.min_lr) * phi_factor

# ===================================================================
# TRAINER
# ===================================================================

class CosmicTrainer:
    """Complete training system for 12D CST Transformer"""
    
    def __init__(
        self,
        model_config: CosmicConfig,
        train_config: 'TrainingConfig',
        rank: int = 0,
        world_size: int = 1
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.rank = rank
        self.world_size = world_size
        self.is_master = (rank == 0)
        
        # Setup device
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)
        
        # Setup dtype
        self.dtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16
        }[train_config.dtype]
        
        # Create model
        print(f"[RANK {rank}] Initializing 12D CST Transformer...")
        self.model = CosmicSynapseTransformer(model_config)
        self.model.to(self.device)
        
        # Wrap in DDP if distributed
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])
        
        # Compile model (PyTorch 2.0+)
        if train_config.compile and hasattr(torch, 'compile'):
            print(f"[RANK {rank}] Compiling model...")
            self.model = torch.compile(self.model)
        
        # Optimizer
        self.optimizer = self.configure_optimizer()
        
        # GradScaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=(train_config.dtype == 'float16'))
        
        # Logging
        self.setup_logging()
        
        # State
        self.iter_num = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        if self.is_master:
            Path(train_config.ckpt_dir).mkdir(parents=True, exist_ok=True)
    
    def configure_optimizer(self) -> torch.optim.Optimizer:
        """
        Configure AdamW optimizer with weight decay.
        Separate params that should/shouldn't have weight decay.
        """
        # Separate decay/no_decay params
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # No decay for biases and layer norms
                if 'bias' in name or 'ln' in name or 'LayerNorm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.train_config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        # Use fused AdamW if available (faster)
        use_fused = (self.device != 'cpu') and ('fused' in torch.optim.AdamW.__doc__)
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.train_config.learning_rate,
            betas=(self.train_config.beta1, self.train_config.beta2),
            fused=use_fused
        )
        
        print(f"[OPTIMIZER] Using {'fused' if use_fused else 'standard'} AdamW")
        print(f"[OPTIMIZER] Decay params: {len(decay_params):,}, No-decay: {len(no_decay_params):,}")
        
        return optimizer
    
    def setup_logging(self) -> None:
        """Setup wandb and/or tensorboard logging"""
        self.writer = None
        self.wandb_run = None
        
        if not self.is_master:
            return
        
        # Wandb
        if HAS_WANDB and self.train_config.wandb_project:
            self.wandb_run = wandb.init(
                project=self.train_config.wandb_project,
                name=self.train_config.wandb_run_name,
                config={
                    **asdict(self.model_config),
                    **asdict(self.train_config)
                }
            )
            print("[LOGGING] Wandb initialized")
        
        # TensorBoard
        if HAS_TENSORBOARD:
            log_dir = Path(self.train_config.ckpt_dir) / "logs"
            log_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            print(f"[LOGGING] TensorBoard logging to {log_dir}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to wandb and/or tensorboard"""
        if not self.is_master:
            return
        
        if self.wandb_run:
            wandb.log(metrics, step=step)
        
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint"""
        if not self.is_master:
            return
        
        # Get raw model (unwrap DDP if needed)
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_config': asdict(self.model_config),
            'train_config': asdict(self.train_config),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
        }
        
        torch.save(checkpoint, filepath)
        print(f"[CHECKPOINT] Saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Get raw model
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        raw_model.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"[CHECKPOINT] Loaded from {filepath} (iter {self.iter_num})")
    
    @torch.no_grad()
    def estimate_loss(self, data_loader: DataLoader, max_iters: Optional[int] = None) -> Dict[str, float]:
        """Estimate loss on dataset"""
        self.model.eval()
        losses = []
        x12_values = []
        
        max_iters = max_iters or self.train_config.eval_iters
        
        for i, (x, y) in enumerate(data_loader):
            if i >= max_iters:
                break
            
            x, y = x.to(self.device), y.to(self.device)
            
            with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
                logits, loss, metrics = self.model(x, y)
            
            losses.append(loss.item())
            x12_values.append(metrics['x12_final'])
        
        self.model.train()
        
        return {
            'loss': np.mean(losses),
            'x12_mean': np.mean(x12_values),
            'x12_std': np.std(x12_values)
        }
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        print("\n" + "="*60)
        print("STARTING 12D COSMIC SYNAPSE TRANSFORMER TRAINING")
        print("="*60)
        
        self.model.train()
        train_iter = iter(train_loader)
        
        # Training loop
        t0 = time.time()
        running_loss = 0.0
        
        while self.iter_num < self.train_config.max_iters:
            # Learning rate schedule
            lr = get_lr(self.iter_num, self.train_config)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Evaluation
            if self.iter_num % self.train_config.eval_interval == 0:
                if val_loader is not None and self.is_master:
                    val_metrics = self.estimate_loss(val_loader)
                    
                    print(f"\n[EVAL @ iter {self.iter_num}]")
                    print(f"  Val Loss: {val_metrics['loss']:.4f}")
                    print(f"  Val x₁₂: {val_metrics['x12_mean']:.4f} ± {val_metrics['x12_std']:.4f}")
                    
                    self.log_metrics({
                        'val/loss': val_metrics['loss'],
                        'val/x12_mean': val_metrics['x12_mean'],
                        'val/x12_std': val_metrics['x12_std']
                    }, self.iter_num)
                    
                    # Save best model
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.save_checkpoint(
                            os.path.join(self.train_config.ckpt_dir, 'best_model.pt')
                        )
            
            # Checkpointing
            if self.iter_num % self.train_config.save_interval == 0 and self.iter_num > 0:
                self.save_checkpoint(
                    os.path.join(self.train_config.ckpt_dir, f'ckpt_iter_{self.iter_num}.pt')
                )
            
            # Training step
            micro_losses = []
            
            for micro_step in range(self.train_config.gradient_accumulation_steps):
                # Get batch
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass with mixed precision
                with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
                    logits, loss, metrics = self.model(x, y)
                    # Scale loss for gradient accumulation
                    loss = loss / self.train_config.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                micro_losses.append(loss.item())
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_config.grad_clip
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            # Logging
            loss_value = sum(micro_losses)
            running_loss += loss_value
            
            if self.iter_num % self.train_config.log_interval == 0 and self.is_master:
                avg_loss = running_loss / self.train_config.log_interval
                t1 = time.time()
                dt = t1 - t0
                tokens_per_sec = (
                    self.train_config.batch_size *
                    self.train_config.gradient_accumulation_steps *
                    self.model_config.max_seq_len *
                    self.train_config.log_interval *
                    self.world_size
                ) / dt
                
                print(f"iter {self.iter_num:6d} | "
                      f"loss {avg_loss:.4f} | "
                      f"lr {lr:.6f} | "
                      f"x₁₂ {metrics['x12_final']:.4f} | "
                      f"{tokens_per_sec:.0f} tok/s")
                
                self.log_metrics({
                    'train/loss': avg_loss,
                    'train/lr': lr,
                    'train/x12_mean': metrics['x12_final'],
                    'train/x12_std': metrics['x12_std'],
                    'train/tokens_per_sec': tokens_per_sec
                }, self.iter_num)
                
                running_loss = 0.0
                t0 = time.time()
            
            self.iter_num += 1
        
        # Final save
        if self.is_master:
            self.save_checkpoint(
                os.path.join(self.train_config.ckpt_dir, 'final_model.pt')
            )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)

# ===================================================================
# DISTRIBUTED TRAINING SETUP
# ===================================================================

def setup_distributed(rank: int, world_size: int, backend: str = 'nccl') -> None:
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_distributed() -> None:
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train_distributed(rank: int, world_size: int, model_config: CosmicConfig, train_config: 'TrainingConfig') -> None:
    """Distributed training worker"""
    setup_distributed(rank, world_size, train_config.backend)
    
    # Create trainer
    trainer = CosmicTrainer(model_config, train_config, rank, world_size)
    
    # Create datasets
    train_dataset = TextDataset(
        train_config.train_data_path,
        model_config.max_seq_len
    )
    
    val_dataset = TextDataset(
        train_config.val_data_path,
        model_config.max_seq_len
    ) if train_config.val_data_path else None
    
    # Create samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if val_dataset else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    ) if val_dataset else None
    
    # Train
    trainer.train(train_loader, val_loader)
    
    cleanup_distributed()

# ===================================================================
# MAIN TRAINING SCRIPT
# ===================================================================

def main() -> None:
    """Main training entry point"""
    
    # Model configuration
    model_config = CosmicConfig(
        vocab_size=50257,
        max_seq_len=1024,
        d_model=768,
        n_layers=12,
        n_heads=12,
        dropout=0.1,
    )
    
    # Training configuration
    train_config = TrainingConfig(
        train_data_path="data/train.bin",
        val_data_path="data/val.bin",
        batch_size=8,
        gradient_accumulation_steps=4,
        max_iters=100000,
        learning_rate=3e-4 * PHI_INV,
        ckpt_dir="checkpoints/12d_cst",
        wandb_project="12d-cst-transformer",
        wandb_run_name="phi-harmonic-768d-12L",
    )
    
    # Check if distributed
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        print(f"[DISTRIBUTED] Training on {world_size} GPUs")
        mp.spawn(
            train_distributed,
            args=(world_size, model_config, train_config),
            nprocs=world_size,
            join=True
        )
    else:
        print("[SINGLE GPU] Training on 1 GPU")
        trainer = CosmicTrainer(model_config, train_config)
        
        # Create datasets
        train_dataset = TextDataset(
            train_config.train_data_path,
            model_config.max_seq_len
        )
        
        val_dataset = TextDataset(
            train_config.val_data_path,
            model_config.max_seq_len
        ) if train_config.val_data_path else None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ) if val_dataset else None
        
        # Train
        trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
