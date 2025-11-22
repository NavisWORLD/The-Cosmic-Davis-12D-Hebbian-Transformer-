"""
AUTONOMOUS STUDY SESSION
========================
"I will be gone for a couple hours..."

This script runs a continuous, self-directed learning loop for 12D and 42D models.
It covers a UNIVERSAL CURRICULUM of diverse topics:
- Literature & Fiction
- Science & Physics
- Philosophy & Logic
- History & Politics
- Technology & Code

Features:
- Auto-downloading of texts
- Continuous training
- Progress logging
- Checkpoint saving
"""

import sys
import os
import time
import random
import torch
import datetime
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parents[2] / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(Path(__file__).parents[0])) 

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from cosmic_synapse.data.web_reader import WebCurriculum
from cosmic_synapse.training.unified_trainer import UnifiedCosmicTrainer
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig

# ===================================================================
# THE UNIVERSAL CURRICULUM
# ===================================================================

TOPICS = {
    "LITERATURE": [
        "https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
        "https://www.gutenberg.org/cache/epub/84/pg84.txt",  # Frankenstein
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
    ],
    "PHILOSOPHY": [
        "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",  # The Republic (Plato)
        "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",  # Beyond Good and Evil (Nietzsche)
        "https://www.gutenberg.org/cache/epub/74/pg74.txt",  # Tom Sawyer (Mark Twain) - acting as 'Wisdom'
    ],
    "SCIENCE": [
        "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",  # Origin of Species (Darwin)
        "https://www.gutenberg.org/cache/epub/30155/pg30155.txt", # Relativity (Einstein)
    ],
    "HISTORY": [
        "https://www.gutenberg.org/cache/epub/5/pg5.txt",  # US Constitution
        "https://www.gutenberg.org/cache/epub/1080/pg1080.txt", # A Modest Proposal
    ],
    "TECHNOLOGY": [
        "https://docs.python.org/3/text/tutorial.txt", # Python Tutorial (Simulated URL, we will use a real text dump if possible, or a reliable raw text source)
        # Since official docs aren't easily raw-text accessible via simple GET without parsing, 
        # we will use Project Gutenberg's technical section or similar reliable raw text.
        # For now, using a reliable placeholder for "The Art of War" (Strategy/Logic) and "The Prince" (Political Science/Logic)
        # which act as proxies for algorithmic thinking until we have a raw code dump.
        # ACTUALLY, let's use a raw GitHub file for a real coding example.
        "https://raw.githubusercontent.com/python/cpython/main/README.rst", # Python Readme
        "https://raw.githubusercontent.com/torvalds/linux/master/README", # Linux Readme
        "https://www.gutenberg.org/cache/epub/3600/pg3600.txt", # The Art of War (Strategy/Logic)
    ]
}

def setup_logging():
    log_dir = Path("study_session_logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir

def log_progress(log_file, message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with open(log_file, "a", encoding='utf-8') as f:
        f.write(entry + "\n")

def run_autonomous_study():
    log_dir = setup_logging()
    log_file = log_dir / "study_log.txt"
    
    log_progress(log_file, "="*60)
    log_progress(log_file, "STARTING AUTONOMOUS STUDY SESSION")
    log_progress(log_file, "Curriculum: Literature, Philosophy, Science, History")
    log_progress(log_file, "="*60)
    
    # 1. Setup Tokenizer (FULL VOCABULARY MODE)
    gen = SyntheticDataGenerator()
    
    vocab_path = log_dir / "vocab.txt"
    
    if vocab_path.exists():
        log_progress(log_file, "Loading existing vocabulary...")
        # Load vocab logic would go here, but for simplicity in this script we'll rebuild or assume
        # actually, let's just rebuild it if it's missing, or load it.
        # For now, let's ALWAYS rebuild to be safe and ensure it matches the curriculum.
        pass

    log_progress(log_file, "BUILDING FULL VOCABULARY (Pre-scanning books)...")
    all_text_samples = []
    
    # Scan a few books from each category to build a representative vocab
    scan_urls = [
        TOPICS["LITERATURE"][0], # Alice
        TOPICS["SCIENCE"][0],    # Origin of Species
        TOPICS["PHILOSOPHY"][0], # Republic
        TOPICS["HISTORY"][0]     # Constitution
    ]
    
    for url in scan_urls:
        log_progress(log_file, f"Scanning for vocab: {url}")
        try:
            reader = WebCurriculum([url])
            for _, content in reader.read_stream():
                all_text_samples.append(content)
        except Exception as e:
            log_progress(log_file, f"Failed to scan {url}: {e}")

    # Combine and build vocab
    full_text = " ".join(all_text_samples)
    # Simple whitespace splitting for "words"
    tokens = full_text.split()
    
    # Build vocab up to 30,000 words
    gen.build_vocabulary(tokens, max_vocab_size=30000)
    vocab_size = gen.vocab_size
    
    # Save vocab for teacher_eval.py
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word, idx in gen.vocab.items():
            f.write(f"{word}\t{idx}\n")
            
    log_progress(log_file, f"FULL VOCABULARY INITIALIZED: {vocab_size} tokens")
    log_progress(log_file, f"Vocabulary saved to {vocab_path}")
    
    # 2. Initialize Models
    # We use slightly larger models for this long run
    # Increased d_model to 256 for better capacity with large vocab
    config_12d = CosmicConfig(vocab_size=vocab_size, d_model=256, n_layers=6, n_heads=8)
    model_12d = CosmicSynapseTransformer(config_12d)
    trainer_12d = UnifiedCosmicTrainer(model_12d, config_12d, model_type="12D")
    
    config_42d = HyperConfig(vocab_size=vocab_size, d_model=256, n_layers=6, n_heads=8)
    model_42d = HyperCosmicTransformer(config_42d)
    trainer_42d = UnifiedCosmicTrainer(model_42d, config_42d, model_type="42D")
    
    log_progress(log_file, "Models initialized (256 dim, 6 layers, 8 heads)")
    
    # 3. Study Loop
    iteration = 0
    
    try:
        while True:
            iteration += 1
            
            # Pick a random topic
            topic = random.choice(list(TOPICS.keys()))
            url = random.choice(TOPICS[topic])
            
            log_progress(log_file, f"\n--- SESSION {iteration}: {topic} ---")
            log_progress(log_file, f"Source: {url}")
            
            # Read content
            reader = WebCurriculum([url])
            
            for _, content in reader.read_stream():
                log_progress(log_file, f"Content loaded ({len(content)} chars). Studying...")
                
                # Train 12D
                res_12d = trainer_12d.train_on_text(content, gen, seq_len=128, batch_size=16)
                loss_12d = res_12d['loss']
                
                # Train 42D
                res_42d = trainer_42d.train_on_text(content, gen, seq_len=128, batch_size=16)
                loss_42d = res_42d['loss']
                
                # Log results
                diff = loss_12d - loss_42d
                winner = "42D" if diff > 0 else "12D"
                
                log_progress(log_file, f"12D Loss: {loss_12d:.4f}")
                log_progress(log_file, f"42D Loss: {loss_42d:.4f}")
                log_progress(log_file, f"Winner: {winner} (Advantage: {abs(diff):.4f})")
                
                # Save checkpoints
                torch.save(model_12d.state_dict(), log_dir / "model_12d_latest.pt")
                torch.save(model_42d.state_dict(), log_dir / "model_42d_latest.pt")
                log_progress(log_file, "Checkpoints saved.")
                
            # Small break between books to simulate "pondering" (and let CPU cool slightly)
            time.sleep(2)
            
    except KeyboardInterrupt:
        log_progress(log_file, "Study session interrupted by user.")
    except Exception as e:
        log_progress(log_file, f"CRITICAL ERROR: {e}")
        raise e

if __name__ == "__main__":
    run_autonomous_study()
