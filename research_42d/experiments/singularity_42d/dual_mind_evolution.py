"""
DUAL MIND EVOLUTION SYSTEM
==========================
12D (Past/Grounded) ⇄ 42D (Future/Abstract) → Present Understanding

This system allows the two models to:
- Learn independently
- Debate and collaborate
- Form consensus understanding
- Save infinite memory checkpoints
"""

import torch
import torch.nn as nn
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

# Add paths
sys.path.append(str(Path(__file__).parents[2] / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(Path(__file__).parents[0]))

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig


class InfiniteMemorySystem:
    """
    4D Memory System with auto-save checkpoints
    Implements incremental saves to simulate brain memory
    """
    
    def __init__(self, base_dir="memory_vault"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Memory layers (like brain memory consolidation)
        self.working_memory = []  # Latest experiences
        self.short_term = []       # Recent sessions
        self.long_term_index = {}  # Indexed memories
        
        self.checkpoint_counter = 0
        self.auto_save_interval = 100  # Save every 100 iterations
        
    def create_checkpoint(self, models_state: Dict, metadata: Dict) -> str:
        """Create a timestamped checkpoint"""
        timestamp = datetime.now().isoformat().replace(":", "-")
        checkpoint_id = f"checkpoint_{self.checkpoint_counter:06d}_{timestamp}"
        
        checkpoint_path = self.base_dir / f"{checkpoint_id}.pt"
        
        # Save model states
        torch.save({
            'models': models_state,
            'metadata': metadata,
            'checkpoint_id': checkpoint_id,
            'timestamp': timestamp,
            'iteration': self.checkpoint_counter
        }, checkpoint_path)
        
        # Update index
        self.long_term_index[checkpoint_id] = {
            'path': str(checkpoint_path),
            'timestamp': timestamp,
            'metadata': metadata
        }
        
        self.checkpoint_counter += 1
        return checkpoint_id
    
    def get_checkpoint_history(self, limit=10):
        """Get recent checkpoints"""
        sorted_ids = sorted(self.long_term_index.keys(), reverse=True)
        return [(id, self.long_term_index[id]) for id in sorted_ids[:limit]]
    
    def load_checkpoint(self, checkpoint_id: str):
        """Load a specific checkpoint"""
        if checkpoint_id in self.long_term_index:
            path = self.long_term_index[checkpoint_id]['path']
            return torch.load(path, map_location='cpu')
        return None


class DualMindDebateSystem:
    """
    The Debate System: 12D vs 42D
    - 12D represents grounded, past knowledge
    - 42D represents abstract, future potential
    - They debate to form present understanding
    """
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        
        # Initialize both minds
        config_12d = CosmicConfig(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=6,
            n_heads=8
        )
        self.mind_12d = CosmicSynapseTransformer(config_12d)
        
        config_42d = HyperConfig(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=6,
            n_heads=8
        )
        self.mind_42d = HyperCosmicTransformer(config_42d)
        
        # Debate history
        self.debate_log = []
        
    def generate_response(self, model, input_ids, temperature=0.8, max_tokens=20):
        """Generate response from a model using forward pass"""
        model.eval()
        with torch.no_grad():
            # Get initial logits
            logits, _, _ = model(input_ids)
            
            # Sample from logits
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Start with input and add token
            output = torch.cat([input_ids, next_token], dim=1)
            
            # Generate more tokens
            for _ in range(max_tokens - 1):
                logits, _, _ = model(output)
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                output = torch.cat([output, next_token], dim=1)
            
        return output
    
    def debate(self, prompt_ids: torch.Tensor, rounds: int = 3) -> Dict:
        """
        Conduct a debate between 12D and 42D
        
        Returns:
            Dictionary with both perspectives and consensus
        """
        debate_record = {
            'prompt': prompt_ids.tolist(),
            'rounds': []
        }
        
        current_context_12d = prompt_ids
        current_context_42d = prompt_ids
        
        try:
            for round_num in range(rounds):
                # 12D's turn (Grounded perspective)
                response_12d = self.generate_response(
                    self.mind_12d,
                    current_context_12d,
                    temperature=0.7  # More conservative
                )
                
                # 42D's turn (Abstract perspective)
                response_42d = self.generate_response(
                    self.mind_42d,
                    current_context_42d,
                    temperature=0.9  # More creative
                )
                
                # Record round
                debate_record['rounds'].append({
                    '12d_response': response_12d.tolist(),
                    '42d_response': response_42d.tolist(),
                    'round': round_num + 1
                })
                
                # Update contexts - use last part of responses
                # Truncate to avoid growing too large
                max_context_len = 50
                if response_42d.size(1) > max_context_len:
                    current_context_12d = response_42d[:, -max_context_len:]
                else:
                    current_context_12d = response_42d
                    
                if response_12d.size(1) > max_context_len:
                    current_context_42d = response_12d[:, -max_context_len:]
                else:
                    current_context_42d = response_12d
            
            # Final consensus (combine both final outputs)
            consensus = self._form_consensus(response_12d, response_42d)
            debate_record['consensus'] = consensus.tolist()
            
            self.debate_log.append(debate_record)
            
        except Exception as e:
            print(f"Error during debate: {e}")
            debate_record['error'] = str(e)
        
        return debate_record
    
    def _form_consensus(self, output_12d, output_42d):
        """
        Form consensus by merging both perspectives
        Simple approach: interleave or average logits
        """
        # For now, return 12D's output (can be enhanced)
        return output_12d
    
    def get_debate_summary(self):
        """Get summary of recent debates"""
        return {
            'total_debates': len(self.debate_log),
            'recent': self.debate_log[-5:] if self.debate_log else []
        }


class EvolutionaryLearningLoop:
    """
    Main training loop with debate and evolution
    """
    
    def __init__(self, vocab_size: int):
        self.dual_mind = DualMindDebateSystem(vocab_size)
        self.memory = InfiniteMemorySystem()
        self.iteration = 0
        
    def train_iteration(self, input_ids: torch.Tensor, target_ids: torch.Tensor):
        """
        Single training iteration with debate
        """
        # 1. Individual learning
        self.dual_mind.mind_12d.train()
        self.dual_mind.mind_42d.train()
        
        # Train both models separately
        logits_12d, loss_12d, _ = self.dual_mind.mind_12d(input_ids, targets=target_ids)
        logits_42d, loss_42d, _ = self.dual_mind.mind_42d(input_ids, targets=target_ids)
        
        # 2. Debate phase (every 10 iterations)
        debate_result = None
        if self.iteration % 10 == 0:
            self.dual_mind.mind_12d.eval()
            self.dual_mind.mind_42d.eval()
            debate_result = self.dual_mind.debate(input_ids[:1, :10], rounds=2)
        
        # 3. Auto-checkpoint (every 100 iterations)
        if self.iteration % self.memory.auto_save_interval == 0:
            checkpoint_id = self.memory.create_checkpoint(
                models_state={
                    '12d': self.dual_mind.mind_12d.state_dict(),
                    '42d': self.dual_mind.mind_42d.state_dict()
                },
                metadata={
                    'iteration': self.iteration,
                    'loss_12d': loss_12d.item(),
                    'loss_42d': loss_42d.item(),
                    'total_debates': len(self.dual_mind.debate_log)
                }
            )
            print(f"💾 Checkpoint saved: {checkpoint_id}")
        
        self.iteration += 1
        
        return {
            'loss_12d': loss_12d.item(),
            'loss_42d': loss_42d.item(),
            'debate': debate_result,
            'iteration': self.iteration
        }
    
    def save_state(self):
        """Manual save"""
        return self.memory.create_checkpoint(
            models_state={
                '12d': self.dual_mind.mind_12d.state_dict(),
                '42d': self.dual_mind.mind_42d.state_dict()
            },
            metadata={
                'iteration': self.iteration,
                'debates': len(self.dual_mind.debate_log)
            }
        )


if __name__ == "__main__":
    print("🧠 Dual Mind Evolution System Initialized")
    print("12D (Past) ⇄ 42D (Future) → Present")
