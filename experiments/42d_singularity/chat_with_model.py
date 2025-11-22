"""
CHAT WITH 42D MODEL
===================
Interactive chat interface for the 42D Hyper-Cosmic Transformer.
Talk to the model while it learns!
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

def load_vocab(vocab_path):
    gen = SyntheticDataGenerator()
    if not vocab_path.exists():
        print("❌ No vocabulary file found! Is the study session running?")
        return None
        
    print(f"[SYSTEM] Loading vocabulary from {vocab_path}...")
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
        print("❌ No checkpoint found! Is the study session running?")
        return None

    print(f"[SYSTEM] Loading brain state from {checkpoint_path}...")
    try:
        # Initialize empty model
        config = HyperConfig(vocab_size=vocab_size, d_model=256, n_layers=6, n_heads=8)
        model = HyperCosmicTransformer(config)
        
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def chat_loop():
    print("\n" + "="*60)
    print("🤖 42D HYPER-COSMIC CHAT INTERFACE")
    print("   Type 'exit' to quit.")
    print("="*60)
    
    vocab_path = Path("study_session_logs/vocab.txt")
    checkpoint_path = Path("study_session_logs/model_42d_latest.pt")
    
    gen = load_vocab(vocab_path)
    if not gen: return
    
    model = load_model(checkpoint_path, gen.vocab_size)
    if not model: return
    
    # Reverse vocab for decoding
    id_to_word = {v: k for k, v in gen.vocab.items()}
    
    history = []
    
    while True:
        try:
            user_input = input("\n👤 YOU: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            # Reload model periodically to get latest learning? 
            # For now, let's keep it static per session for stability, 
            # or user can restart script to get latest brain.
            
            print("🤖 42D: ", end="", flush=True)
            
            # Encode input
            input_ids = []
            for word in user_input.lower().split():
                if word in gen.vocab:
                    input_ids.append(gen.vocab[word])
                else:
                    # Unknown word handling
                    pass
            
            if not input_ids:
                input_ids = [0] # Fallback
                
            context = torch.tensor([input_ids], dtype=torch.long)
            
            # Generate
            with torch.no_grad():
                output_ids = model.generate(context, max_new_tokens=50, temperature=0.7)
                
            # Decode response
            response_tokens = []
            for idx in output_ids[0].tolist()[len(input_ids):]: # Only new tokens
                if idx in id_to_word:
                    word = id_to_word[idx]
                    response_tokens.append(word)
                    print(word + " ", end="", flush=True)
                    time.sleep(0.05) # Typing effect
            
            print() # Newline
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    chat_loop()
