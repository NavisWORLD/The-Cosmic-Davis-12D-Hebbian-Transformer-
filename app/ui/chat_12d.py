"""
12D COSMIC CONSOLE (PRODUCTION)
===============================
Professional-grade interface for the 12D Cosmic Synapse Transformer.
"""

import sys
import torch
import time
from pathlib import Path

# Add paths (Adjusted for production_12d/ location)
# Root is one level up
ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))

from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig

def load_vocab(vocab_path):
    gen = SyntheticDataGenerator()
    if not vocab_path.exists():
        print(f"❌ Vocabulary not found at {vocab_path}")
        return None
        
    print(f"[SYSTEM] Loading vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word, idx = parts
                gen.vocab[word] = int(idx)
    gen.vocab_size = len(gen.vocab)
    return gen

def load_model(checkpoint_path, vocab_size):
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return None

    print(f"[SYSTEM] Loading 12D Model...")
    try:
        # Initialize empty model
        config = CosmicConfig(vocab_size=vocab_size, d_model=256, n_layers=6, n_heads=8)
        model = CosmicSynapseTransformer(config)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def chat_loop():
    print("\n" + "="*60)
    print("🌌 12D COSMIC CONSOLE (PRODUCTION)")
    print("   Type 'exit' to quit.")
    print("="*60)
    
    # In production, we might look in saved_models, but for now we link to the live study logs
    # to allow monitoring the ongoing training.
    vocab_path = ROOT_DIR / "study_session_logs" / "vocab.txt"
    vocab_path = ROOT_DIR / "study_session_logs" / "vocab.txt"
    
    # Find the latest checkpoint
    checkpoint_dir = ROOT_DIR / "study_session_logs" / "checkpoints"
    checkpoint_path = None
    
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("model_12d_iter_*.pt"))
        if checkpoints:
            # Sort by iteration number
            checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
            checkpoint_path = checkpoints[-1]
            print(f"[SYSTEM] Found latest checkpoint: {checkpoint_path.name}")
    
    if not checkpoint_path:
        # Fallback to legacy path if no iteration checkpoints found
        checkpoint_path = ROOT_DIR / "study_session_logs" / "model_12d_latest.pt"
    
    gen = load_vocab(vocab_path)
    if not gen: return
    
    model = load_model(checkpoint_path, gen.vocab_size)
    if not model: return
    
    # Reverse vocab for decoding
    id_to_word = {v: k for k, v in gen.vocab.items()}
    
    while True:
        try:
            user_input = input("\n👤 YOU: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            print("🌌 12D: ", end="", flush=True)
            
            # Encode input
            input_ids = []
            for word in user_input.lower().split():
                if word in gen.vocab:
                    input_ids.append(gen.vocab[word])
                else:
                    pass
            
            if not input_ids:
                input_ids = [0] # Fallback
                
            context = torch.tensor([input_ids], dtype=torch.long)
            
            # Generate
            with torch.no_grad():
                output_ids = model.generate(context, max_new_tokens=50, temperature=0.7)
                
            # Decode response
            for idx in output_ids[0].tolist()[len(input_ids):]: # Only new tokens
                if idx in id_to_word:
                    word = id_to_word[idx]
                    print(word + " ", end="", flush=True)
                    time.sleep(0.05) # Typing effect
            
            print() # Newline
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    chat_loop()
