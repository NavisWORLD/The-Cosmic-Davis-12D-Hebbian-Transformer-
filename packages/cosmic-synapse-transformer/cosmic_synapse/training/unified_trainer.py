"""
UNIFIED COSMIC TRAINER
======================
A unified training pipeline for 12D and 42D Cosmic Synapse models.
Handles model-specific return signatures and manages continuous learning state.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional, Dict, Any, List, Union

class UnifiedCosmicTrainer:
    """
    Trainer that supports both 12D (Scalar) and 42D (Vector) architectures.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 config: Any, 
                 model_type: str = "12D",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.config = config
        self.model_type = model_type
        self.device = device
        
        # φ-scaled learning rate
        # 42D models might need slightly lower LR due to complexity
        base_lr = 3e-4
        phi_inv = 0.61803398875
        self.lr = base_lr * phi_inv
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        print(f"[UNIFIED TRAINER] Initialized for {model_type} model on {device}")

    def train_on_text(self, text: str, tokenizer, seq_len: int = 64, batch_size: int = 32) -> Dict[str, float]:
        """
        Train the model on a specific text block (e.g., a web page).
        """
        self.model.train()
        
        # Tokenize
        # Simple character-level fallback if no tokenizer provided, 
        # but ideally we use the model's tokenizer
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text)
        else:
            # Fallback for testing
            tokens = [ord(c) % self.config.vocab_size for c in text]
            
        if len(tokens) <= seq_len + 1:
            return {'loss': 0.0, 'batches': 0}
            
        # Create batches
        data = torch.tensor(tokens, dtype=torch.long)
        n_batches = (len(data) - 1) // (batch_size * seq_len)
        
        if n_batches == 0:
            # Train on single partial batch
            x = data[:-1].unsqueeze(0)
            y = data[1:].unsqueeze(0)
            if x.size(1) > seq_len:
                x = x[:, :seq_len]
                y = y[:, :seq_len]
            batches = [(x, y)]
        else:
            # Create strided batches
            batches = []
            for i in range(0, len(data) - seq_len - 1, seq_len):
                if len(batches) >= n_batches * batch_size: break
                x = data[i:i+seq_len]
                y = data[i+1:i+seq_len+1]
                batches.append((x, y))
            
            # Group into batch_size
            batched_data = []
            for i in range(0, len(batches), batch_size):
                batch = batches[i:i+batch_size]
                if len(batch) < batch_size: continue
                bx = torch.stack([b[0] for b in batch])
                by = torch.stack([b[1] for b in batch])
                batched_data.append((bx, by))
            batches = batched_data

        total_loss = 0
        metrics_accum = {}
        
        for bx, by in batches:
            bx, by = bx.to(self.device), by.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Unified Forward Pass
            if self.model_type == "12D":
                logits, loss, metrics = self.model(bx, targets=by)
            elif self.model_type == "42D":
                # 42D model signature might differ slightly in future, 
                # but currently follows (logits, loss, metrics)
                logits, loss, metrics = self.model(bx, targets=by)
            else:
                # Vanilla fallback
                logits, loss = self.model(bx, by)
                metrics = {}
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Accumulate metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    metrics_accum[k] = metrics_accum.get(k, 0) + v
        
        avg_loss = total_loss / max(1, len(batches))
        result = {'loss': avg_loss, 'batches': len(batches)}
        
        # Add averaged metrics
        for k, v in metrics_accum.items():
            result[k] = v / max(1, len(batches))
            
        return result
