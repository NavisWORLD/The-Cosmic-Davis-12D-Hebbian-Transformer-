"""
COSMIC DAVIS - RAM-AWARE EVOLUTIONARY TUNER
============================================
Memory-optimized instruction tuning. 
Freezes most weights to focus learning on the "personality" layers.
"""

import sys
import os
import time
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import GPT2Tokenizer

# Path Setup
ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig

def run_low_mem_tuning():
    print("="*60)
    print("🌌 COSMIC DAVIS - FINAL POLISH (30K VOCAB)")
    print("="*60)

    # 1. Load Custom Vocab (from training)
    from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
    gen = SyntheticDataGenerator()
    vocab_path = Path("study_session_logs/vocab.txt")
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    gen.vocab[parts[0]] = int(parts[1])
        gen.vocab_size = len(gen.vocab)
    else:
        gen.vocab_size = 30000 
        
    print(f"Vocab size: {gen.vocab_size}")
    
    # 2. Setup Config (must match 5-day trained brain)
    config = CosmicConfig(
        vocab_size=gen.vocab_size,
        d_model=256, 
        n_layers=6,
        n_heads=8
    )
    model = CosmicSynapseTransformer(config)
    
    # 3. Load the 5-Day Trained Brain
    brain_path = Path("study_session_logs/model_12d_latest.pt")
    if brain_path.exists():
        print(f"Loading 5-day trained brain from {brain_path}")
        model.load_state_dict(torch.load(brain_path, map_location='cpu'))
        print("✅ Weights loaded perfectly.")
    
    # 4. RAM-Aware: Allow all parameters now that dimensions match
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 5. Load Instruction Data
    seed_path = Path("study_session_logs/conversation_seed.txt")
    conversation_pairs = []
    if seed_path.exists():
        lines = seed_path.read_text(encoding='utf-8').split('\n')
        curr_q = ""
        for line in lines:
            if line.startswith('Q: '): curr_q = line[3:]
            elif line.startswith('A: ') and curr_q:
                conversation_pairs.append((curr_q, line[3:]))
                curr_q = ""

    print(f"Loaded {len(conversation_pairs)} Q&A pairs. Starting polish...")

    # 6. Tuning Loop
    for epoch in range(3):
        random.shuffle(conversation_pairs)
        total_loss = 0
        for i, (q, a) in enumerate(conversation_pairs):
            text = f"<|human|>\n{q}\n<|assistant|>\n{a}<|endoftext|>"
            
            # Use gen for custom 30k tokenization
            token_list = []
            for word in text.split():
                token_list.append(gen.vocab.get(word, gen.vocab.get("<UNK>", 1)))
            
            if len(token_list) < 2: continue
            
            tokens = torch.tensor([token_list], dtype=torch.long)
            
            x = tokens[:, :-1]
            y = tokens[:, 1:]
            
            logits, loss, metrics, _ = model(x, targets=y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i+1) % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {i+1} | Loss: {total_loss/(i+1):.4f}")

    # 7. Save Polished Brain
    polished_path = Path("study_session_logs/polished_12d_brain.pt")
    torch.save(model.state_dict(), polished_path)
    print(f"✅ Success! Polished brain saved to {polished_path}")

if __name__ == "__main__":
    run_low_mem_tuning()
