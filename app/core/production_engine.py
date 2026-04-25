"""
PRODUCTION COSMIC DAVIS LEARNING ENGINE
========================================
Optimized training loop for maximum learning efficiency.

This is the main training engine that:
1. Trains on diverse curriculum (books, news, conversations)
2. Uses gradient accumulation for effective larger batches
3. Implements cosine annealing learning rate schedule
4. Prioritizes conversation training for better chat
5. Tracks perplexity and saves best checkpoints
6. Broadcasts state to the Neural Monitor

Author: Cosmic Davis AI System
"""

import sys
import os
import time
import random
import json
import math
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(ROOT_DIR / "app" / "modules" / "dual_mind"))
sys.path.append(str(ROOT_DIR / "app" / "modules" / "audio"))

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from cosmic_synapse.data.web_reader import WebCurriculum
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig
from multimodal_fusion import UnifiedMultimodalSystem

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHI = 1.618033988749895

# Weighted curriculum - conversations are prioritized
CURRICULUM = {
    "CONVERSATION": {
        "weight": 5,  # 5x more likely to be selected
        "sources": ["LOCAL_SEED", "LOCAL_LOG"]
    },
    "LITERATURE": {
        "weight": 2,
        "sources": [
            "https://www.gutenberg.org/files/11/11-0.txt",
            "https://www.gutenberg.org/cache/epub/84/pg84.txt",
        ]
    },
    "PHILOSOPHY": {
        "weight": 2,
        "sources": [
            "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
        ]
    },
    "SCIENCE": {
        "weight": 2,
        "sources": [
            "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
        ]
    },
    "NEWS": {
        "weight": 1,
        "sources": [
            "https://text.npr.org/",
        ]
    },
    "SELF": {
        "weight": 1,
        "sources": ["LOCAL_SYSTEM"]
    }
}

# ---------------------------------------------------------------------------
# Learning Rate Scheduler
# ---------------------------------------------------------------------------
class CosineLRScheduler:
    """Cosine annealing with warmup."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# ---------------------------------------------------------------------------
# Production Trainer
# ---------------------------------------------------------------------------
class ProductionTrainer:
    """
    Optimized trainer for both 12D and 42D models.
    """
    
    def __init__(self, model, config, model_type: str = "12D", 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.model_type = model_type
        self.device = device
        
        # Optimized hyperparameters
        self.base_lr = 3e-4 * (1 / PHI)  # φ-scaled
        
        # Use AdamW with better defaults
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.base_lr,
            betas=(0.9, 0.98),  # Slightly higher β2 for stability
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Scheduler for better convergence
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            warmup_steps=100,
            total_steps=10000,
            min_lr=1e-6
        )
        
        # Gradient accumulation for effective larger batches
        self.gradient_accumulation_steps = 4
        self.accumulated_steps = 0
        
        # Best model tracking
        self.best_loss = float('inf')
        
        print(f"[PRODUCTION TRAINER] {model_type} initialized on {device}")
        print(f"[PRODUCTION TRAINER] Base LR: {self.base_lr:.6f}, Grad Accum: {self.gradient_accumulation_steps}")
    
    def train_on_text(self, text: str, tokenizer, seq_len: int = 128, 
                      batch_size: int = 16, epochs: int = 1) -> Dict[str, float]:
        """Train on a text block with gradient accumulation."""
        self.model.train()
        
        # Tokenize
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text)
        else:
            tokens = [ord(c) % self.config.vocab_size for c in text]
        
        if len(tokens) <= seq_len + 1:
            return {'loss': 0.0, 'perplexity': 0.0, 'batches': 0}
        
        # Create dataset
        data = torch.tensor(tokens, dtype=torch.long)
        
        # Create strided examples
        examples = []
        for i in range(0, len(data) - seq_len - 1, seq_len // 2):  # 50% overlap for more data
            x = data[i:i+seq_len]
            y = data[i+1:i+seq_len+1]
            if len(x) == seq_len and len(y) == seq_len:
                examples.append((x, y))
        
        if not examples:
            return {'loss': 0.0, 'perplexity': 0.0, 'batches': 0}
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        for epoch in range(epochs):
            random.shuffle(examples)
            
            # Create batches
            for i in range(0, len(examples), batch_size):
                batch = examples[i:i+batch_size]
                if len(batch) < 2:
                    continue
                
                bx = torch.stack([b[0] for b in batch]).to(self.device)
                by = torch.stack([b[1] for b in batch]).to(self.device)
                
                # Forward pass
                logits, loss, metrics = self.model(bx, targets=by)
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                self.accumulated_steps += 1
                total_loss += loss.item() * self.gradient_accumulation_steps
                total_tokens += bx.numel()
                num_batches += 1
                
                # Gradient step
                if self.accumulated_steps >= self.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.accumulated_steps = 0
        
        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_loss, 10))  # Cap to prevent overflow
        
        # Track best
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'batches': num_batches,
            'lr': self.optimizer.param_groups[0]['lr'],
            'best_loss': self.best_loss
        }

# ---------------------------------------------------------------------------
# Content Loader
# ---------------------------------------------------------------------------
def get_local_system_data() -> str:
    """Generate system awareness data."""
    report = []
    now = datetime.now()
    report.append(f"SYSTEM_TIME: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"OS: {platform.system()} {platform.release()}")
    report.append(f"MACHINE: {platform.node()}")
    report.append(f"PROCESSOR: {platform.processor()}")
    return " ".join(report)

def load_conversation_data() -> str:
    """Load conversation training data."""
    seed_path = ROOT_DIR / "study_session_logs" / "conversation_seed.txt"
    log_path = ROOT_DIR / "study_session_logs" / "conversation_log.txt"
    
    text = ""
    if seed_path.exists():
        text += seed_path.read_text(encoding="utf-8")
    if log_path.exists():
        text += "\n" + log_path.read_text(encoding="utf-8")
    
    return text if text else "Q: Hello. A: Hello! I am Cosmic Davis, ready to learn and chat."

def select_topic() -> Tuple[str, str]:
    """Weighted topic selection."""
    topics = []
    weights = []
    
    for topic, data in CURRICULUM.items():
        topics.append(topic)
        weights.append(data['weight'])
    
    total = sum(weights)
    probs = [w/total for w in weights]
    
    topic = random.choices(topics, weights=probs, k=1)[0]
    source = random.choice(CURRICULUM[topic]['sources'])
    
    return topic, source

def load_content(topic: str, source: str) -> str:
    """Load content for training."""
    if topic == "CONVERSATION":
        return load_conversation_data()
    elif topic == "SELF":
        return get_local_system_data()
    else:
        try:
            reader = WebCurriculum([source])
            for _, content in reader.read_stream():
                return content
        except Exception as e:
            print(f"[WARN] Failed to load {source}: {e}")
            return ""
    return ""

# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def run_production_training():
    """Main production training loop."""
    
    log_dir = ROOT_DIR / "study_session_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "training_log.txt"
    
    def log(msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        print(entry)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    
    log("="*60)
    log("🚀 COSMIC DAVIS PRODUCTION TRAINING ENGINE")
    log("="*60)
    
    # Load vocabulary
    vocab_path = log_dir / "vocab.txt"
    gen = SyntheticDataGenerator()
    
    if vocab_path.exists():
        log("Loading vocabulary...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word, idx = parts
                    gen.vocab[word] = int(idx)
        gen.vocab_size = len(gen.vocab)
        log(f"Vocabulary: {gen.vocab_size} tokens")
    else:
        log("Building vocabulary from scratch...")
        # Quick build from available data
        all_text = load_conversation_data()
        gen.build_vocabulary(all_text.split(), max_vocab_size=30000)
        with open(vocab_path, "w", encoding="utf-8") as f:
            for word, idx in gen.vocab.items():
                f.write(f"{word}\t{idx}\n")
        log(f"Built vocabulary: {gen.vocab_size} tokens")
    
    # Initialize models
    log("Initializing 12D Cosmic Synapse Transformer...")
    config_12d = CosmicConfig(vocab_size=gen.vocab_size, d_model=256, n_layers=6, n_heads=8)
    model_12d = CosmicSynapseTransformer(config_12d)
    trainer_12d = ProductionTrainer(model_12d, config_12d, "12D")
    
    log("Initializing 42D Hyper-Cosmic Transformer...")
    config_42d = HyperConfig(vocab_size=gen.vocab_size, d_model=256, n_layers=6, n_heads=8)
    model_42d = HyperCosmicTransformer(config_42d)
    trainer_42d = ProductionTrainer(model_42d, config_42d, "42D")
    
    # Multimodal senses
    senses = UnifiedMultimodalSystem()
    
    # Metrics tracking
    iteration = 0
    total_tokens_trained = 0
    best_perplexity = float('inf')
    
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    log("="*60)
    log("Starting continuous learning loop...")
    log("="*60)
    
    try:
        while True:
            iteration += 1
            
            # Select topic with weighted probability
            topic, source = select_topic()
            log(f"\n--- Iteration {iteration}: {topic} ---")
            
            # Load content
            content = load_content(topic, source)
            if len(content) < 50:
                log(f"[SKIP] Content too short ({len(content)} chars)")
                continue
            
            log(f"Content: {len(content)} chars")
            
            # Emotional modulation
            _, emotion, thought = senses.process_multimodal_input(text=content[:500])
            log(f"Emotion: {emotion.classify_emotion()}")
            
            # Train BOTH models
            # More epochs for conversation data
            epochs = 3 if topic == "CONVERSATION" else 1
            
            result_12d = trainer_12d.train_on_text(
                content, gen, seq_len=128, batch_size=16, epochs=epochs
            )
            result_42d = trainer_42d.train_on_text(
                content, gen, seq_len=128, batch_size=16, epochs=epochs
            )
            
            log(f"12D: Loss={result_12d['loss']:.4f}, PPL={result_12d['perplexity']:.2f}, LR={result_12d['lr']:.2e}")
            log(f"42D: Loss={result_42d['loss']:.4f}, PPL={result_42d['perplexity']:.2f}, LR={result_42d['lr']:.2e}")
            
            total_tokens_trained += result_12d['batches'] * 128 * 16
            
            # Track best
            avg_ppl = (result_12d['perplexity'] + result_42d['perplexity']) / 2
            if avg_ppl < best_perplexity and avg_ppl > 0:
                best_perplexity = avg_ppl
                log(f"🏆 New Best Perplexity: {best_perplexity:.2f}")
            
            # Checkpoint every 50 iterations
            if iteration % 50 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{iteration}.pt"
                torch.save({
                    'iteration': iteration,
                    'model_12d': model_12d.state_dict(),
                    'model_42d': model_42d.state_dict(),
                    'best_perplexity': best_perplexity,
                    'total_tokens': total_tokens_trained
                }, checkpoint_path)
                log(f"💾 Checkpoint saved: {checkpoint_path.name}")
            
            # Update brain state for monitor
            brain_state = {
                "iteration": iteration,
                "topic": topic,
                "loss_12d": result_12d['loss'],
                "loss_42d": result_42d['loss'],
                "perplexity_12d": result_12d['perplexity'],
                "perplexity_42d": result_42d['perplexity'],
                "best_perplexity": best_perplexity,
                "total_tokens": total_tokens_trained,
                "emotion": emotion.classify_emotion(),
                "valence": emotion.valence,
                "arousal": emotion.arousal,
                "current_thought": thought if thought else "Learning...",
                "learning_rate": result_12d['lr'],
                "last_tokens": content.split()[-20:] if content else []
            }
            
            with open(log_dir / "brain_state.json", "w") as f:
                json.dump(brain_state, f, indent=2, default=str)
            
            # Small delay to prevent overload
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        log("\n🛑 Training stopped by user")
        
        # Final save
        final_path = checkpoint_dir / "checkpoint_final.pt"
        torch.save({
            'iteration': iteration,
            'model_12d': model_12d.state_dict(),
            'model_42d': model_42d.state_dict(),
            'best_perplexity': best_perplexity,
            'total_tokens': total_tokens_trained
        }, final_path)
        log(f"💾 Final checkpoint saved: {final_path.name}")
        log(f"📊 Total iterations: {iteration}")
        log(f"📊 Total tokens trained: {total_tokens_trained:,}")
        log(f"📊 Best perplexity: {best_perplexity:.2f}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_production_training()
