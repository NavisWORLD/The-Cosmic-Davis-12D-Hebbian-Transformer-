"""
TRAIN ON WEB: HUMAN-LIKE LEARNING EXPERIMENT
============================================
Trains 12D and 42D models side-by-side on a curriculum of web pages.
Simulates a human reading through a reading list.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parents[2] / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(Path(__file__).parents[0])) # For hyper_cosmic_model

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from cosmic_synapse.data.web_reader import WebCurriculum
from cosmic_synapse.training.unified_trainer import UnifiedCosmicTrainer
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator # For tokenizer

# Import 42D model (assuming it's in the same dir or accessible)
try:
    from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig
except ImportError:
    # Fallback if running from root
    sys.path.append("experiments/42d_singularity")
    from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig

def run_web_training():
    print("="*80)
    print("HUMAN-LIKE WEB TRAINING EXPERIMENT")
    print("Reading real books/articles and training 12D vs 42D")
    print("="*80)
    
    # 1. Setup Tokenizer (using synthetic generator for simplicity, or could use GPT2)
    print("[SETUP] Initializing Tokenizer...")
    gen = SyntheticDataGenerator()
    # Pre-seed vocab with common English words to make it useful for web text
    common_text = "the be to of and a in that have I it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us"
    gen.build_vocabulary(common_text.split())
    vocab_size = gen.vocab_size
    print(f"Vocab Size: {vocab_size}")
    
    # 2. Initialize Models
    print("[SETUP] Initializing Models...")
    
    # 12D Model
    config_12d = CosmicConfig(
        vocab_size=vocab_size,
        d_model=128, n_layers=4, n_heads=4 # Small for speed
    )
    model_12d = CosmicSynapseTransformer(config_12d)
    trainer_12d = UnifiedCosmicTrainer(model_12d, config_12d, model_type="12D")
    
    # 42D Model
    config_42d = HyperConfig(
        vocab_size=vocab_size,
        d_model=128, n_layers=4, n_heads=4
    )
    model_42d = HyperCosmicTransformer(config_42d)
    trainer_42d = UnifiedCosmicTrainer(model_42d, config_42d, model_type="42D")
    
    # 3. Define Curriculum
    # A mix of literature and technical content
    curriculum = WebCurriculum([
        "https://www.gutenberg.org/files/11/11-0.txt", # Alice in Wonderland
        "https://www.gutenberg.org/cache/epub/84/pg84.txt", # Frankenstein
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" # Shakespeare
    ])
    
    # 4. Training Loop
    print("\n[START] Beginning Reading Session...")
    
    total_pages = 0
    
    for url, content in curriculum.read_stream():
        total_pages += 1
        print(f"\n📖 Reading: {url}")
        print(f"   Content Length: {len(content)} chars")
        
        # Train 12D
        print("   Training 12D Model...", end="", flush=True)
        res_12d = trainer_12d.train_on_text(content, gen, seq_len=64, batch_size=16)
        print(f" Loss: {res_12d['loss']:.4f}")
        
        # Train 42D
        print("   Training 42D Model...", end="", flush=True)
        res_42d = trainer_42d.train_on_text(content, gen, seq_len=64, batch_size=16)
        print(f" Loss: {res_42d['loss']:.4f}")
        
        # Compare
        diff = res_12d['loss'] - res_42d['loss']
        winner = "42D" if diff > 0 else "12D"
        print(f"   Winner: {winner} (Delta: {abs(diff):.4f})")
        
    print("\n" + "="*80)
    print("SESSION COMPLETE")
    print(f"Read {total_pages} documents.")
    print("="*80)

if __name__ == "__main__":
    run_web_training()
