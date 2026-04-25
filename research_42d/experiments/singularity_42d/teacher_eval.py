"""
TEACHER EVALUATION INTERFACE
============================
"Test it like a teacher..."

This script loads the LATEST state of the 42D model (from the autonomous study session)
and allows you to ask it questions to test its understanding.

It runs in READ-ONLY mode so it doesn't interfere with the ongoing training.
"""

import sys
import torch
import time
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parents[2] / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(Path(__file__).parents[0])) 

from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig

def run_teacher_eval(prompt_text):
    print("="*60)
    print("TEACHER EVALUATION MODE")
    print(f"Prompt: \"{prompt_text}\"")
    print("="*60)
    
    # 1. Setup Tokenizer (Must match training)
    gen = SyntheticDataGenerator()
    
    vocab_path = Path("study_session_logs/vocab.txt")
    if not vocab_path.exists():
        print("❌ No vocabulary file found! Is the study session running?")
        return
        
    print(f"[TEACHER] Loading vocabulary from {vocab_path}...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word, idx = parts
                gen.vocab[word] = int(idx)
    
    gen.vocab_size = len(gen.vocab)
    vocab_size = gen.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # 2. Load Model
    # Note: Model dimensions must match autonomous_study.py (256 dim, 6 layers, 8 heads)
    checkpoint_path = Path("study_session_logs/model_42d_latest.pt")
    
    if not checkpoint_path.exists():
        print("❌ No checkpoint found! Is the study session running?")
        return

    print(f"[TEACHER] Loading latest brain state from {checkpoint_path}...")
    try:
        # Initialize empty model with NEW dimensions
        config = HyperConfig(vocab_size=vocab_size, d_model=256, n_layers=6, n_heads=8)
        model = HyperCosmicTransformer(config)
        
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval() # Set to evaluation mode
        print("✅ Brain state loaded successfully.")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 3. Generate Response
    print("\n[TEACHER] Generating response...")
    
    # Encode prompt
    prompt_ids = []
    for word in prompt_text.lower().split():
        if word in gen.vocab:
            prompt_ids.append(gen.vocab[word])
        else:
            # Unknown word handling
            pass
            
    if not prompt_ids:
        print("⚠️ Prompt contained no known vocabulary words. Using random seed.")
        prompt_ids = [random.randint(0, vocab_size-1)]
        
    context = torch.tensor([prompt_ids], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(context, max_new_tokens=50, temperature=0.8)
        
    # Decode
    # Create reverse vocab
    id_to_word = {v: k for k, v in gen.vocab.items()}
    output_text = []
    for idx in output_ids[0].tolist():
        if idx in id_to_word:
            output_text.append(id_to_word[idx])
        else:
            output_text.append("<UNK>")
    
    print("\n" + "-"*40)
    print("STUDENT (42D) RESPONSE:")
    print("-" * 40)
    print(" ".join(output_text))
    print("-" * 40)

if __name__ == "__main__":
    if len(sys.path) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "nature of species" # Default prompt
        
    run_teacher_eval(prompt)
