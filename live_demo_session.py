"""
LIVE DEMO SESSION
=================
Replicates the main autonomous evolution loop but restricted to 3 iterations
to verify live operation without an infinite loop.
"""

import sys
import torch
import random
import json
from pathlib import Path
from datetime import datetime

# Add paths
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "packages" / "cosmic-synapse-transformer"))
sys.path.insert(0, str(ROOT / "research_42d" / "experiments" / "singularity_42d"))
sys.path.insert(0, str(ROOT / "research_42d" / "audio_12d"))

from cosmic_synapse.data.web_reader import WebCurriculum
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from cosmic_synapse.training.unified_trainer import UnifiedCosmicTrainer
from dual_mind_evolution import EvolutionaryLearningLoop, DualMindDebateSystem, InfiniteMemorySystem
from multimodal_fusion import UnifiedMultimodalSystem

# Curriculum
TOPICS = {
    "LITERATURE": [
        "https://www.gutenberg.org/files/11/11-0.txt",
    ],
    "PHILOSOPHY": [
        "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
    ],
    "SCIENCE": [
        "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
    ]
}

def log_progress(log_file, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    # We skip writing to file for this demo to avoid cluttering real logs if needed, 
    # but the original script does it. Let's just print.

def run_live_demo():
    print("* STARTING LIVE DUAL MIND DEMO *")
    print("Goal: Run 3 autonomous iterations to prove system is live.")
    
    # Build vocabulary
    print("Building unified vocabulary...")
    gen = SyntheticDataGenerator()
    
    vocab_path = Path("study_session_logs") / "vocab.txt"
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word, idx = parts
                    gen.vocab[word] = int(idx)
        gen.vocab_size = len(gen.vocab)
    else:
        # Quick vocab build
        gen.build_vocabulary(["test", "hello", "world"], max_vocab_size=10000)
    
    print(f"Vocabulary: {gen.vocab_size} tokens")
    
    # Initialize Evolution System
    evolution_loop = EvolutionaryLearningLoop(gen.vocab_size)
    
    # Initialize trainers
    trainer_12d = UnifiedCosmicTrainer(
        evolution_loop.dual_mind.mind_12d,
        evolution_loop.dual_mind.mind_12d.config,
        model_type="12D",
        device='cpu' # Force CPU for safer demo
    )
    
    trainer_42d = UnifiedCosmicTrainer(
        evolution_loop.dual_mind.mind_42d,
        evolution_loop.dual_mind.mind_42d.config,
        model_type="42D",
        device='cpu'
    )
    
    # Multimodal senses
    senses = UnifiedMultimodalSystem()
    base_lr = 3e-4 * 0.618
    
    print("Systems initialized. Beginning evolution cycle (Limit: 3 iterations)...")
    
    iteration = 0
    
    try:
        while iteration < 3:
            iteration += 1
            
            # Pick random topic
            topic = random.choice(list(TOPICS.keys()))
            url = random.choice(TOPICS[topic])
            
            print(f"\n--- ITERATION {iteration}: {topic} ---")
            
            # Read content
            reader = WebCurriculum([url])
            
            for _, content in reader.read_stream():
                print(f"Content: {len(content)} chars")
                
                # Emotion detection
                short_text = content[:500]
                _, emotion, thought = senses.process_multimodal_input(text=short_text)
                mod_params = senses.get_modulated_parameters()
                current_lr = base_lr * mod_params['k']
                
                print(f"Emotion: {emotion.classify_emotion()}")
                print(f"Thought: {thought}")
                
                # Train both minds
                res_12d = trainer_12d.train_on_text(content, gen, seq_len=128, batch_size=4, learning_rate=current_lr)
                res_42d = trainer_42d.train_on_text(content, gen, seq_len=128, batch_size=4, learning_rate=current_lr)
                
                loss_12d = res_12d['loss']
                loss_42d = res_42d['loss']
                
                print(f"✓ 12D Loss: {loss_12d:.4f} | 42D Loss: {loss_42d:.4f}")
                
                # Force a debate on iteration 3 for demo
                if iteration == 3:
                    print("(!) FORCING DEBATE MODE (Demo)")
                    
                    # Create debate prompt from content
                    words = content.split()[:20]
                    debate_ids = torch.tensor([[gen.vocab.get(w.lower(), 0) for w in words]])
                    
                    debate = evolution_loop.dual_mind.debate(debate_ids, rounds=2)
                    print("Debate Output:")
                    for r in debate.get('rounds', []):
                        print(f"  {r['speaker']}: {r['text'][:50]}...")

                break  # One content per iteration
    
    except Exception as e:
        print(f"[X] ERROR: {e}")
        raise e

    print("\n[OK] LIVE DEMO COMPLETE. System is autonomous.")

if __name__ == "__main__":
    run_live_demo()
