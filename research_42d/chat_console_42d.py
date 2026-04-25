"""
42D HYPER CONSOLE (RESEARCH)
============================
Experimental interface for the 42D Hyper-Cosmic Transformer.
Features: Memory, Slash Commands, Rich Visuals, Thinking Mode.
"""

import sys
import torch
import time
import os
from pathlib import Path
from collections import deque

# Add paths
ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(ROOT_DIR / "research_42d" / "experiments" / "singularity_42d")) # For HyperCosmic

from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m" # 42D Color
RESET = "\033[0m"
BOLD = "\033[1m"

class ConversationManager:
    def __init__(self, vocab, max_history_tokens=512):
        self.vocab = vocab
        self.id_to_word = {v: k for k, v in vocab.items()}
        self.max_history = max_history_tokens
        self.history_ids = []
        self.temperature = 0.8 # Slightly higher for 42D creativity
        self.top_k = 50
        
    def encode(self, text):
        ids = []
        for word in text.lower().split():
            if word in self.vocab:
                ids.append(self.vocab[word])
        return ids
        
    def add_user_input(self, text):
        ids = self.encode(text)
        self.history_ids.extend(ids)
        self.trim_history()
        return ids
        
    def add_model_output(self, ids):
        self.history_ids.extend(ids)
        self.trim_history()
        
    def trim_history(self):
        if len(self.history_ids) > self.max_history:
            self.history_ids = self.history_ids[-self.max_history:]
            
    def get_context(self):
        if not self.history_ids:
            return torch.tensor([[0]], dtype=torch.long)
        return torch.tensor([self.history_ids], dtype=torch.long)

    def decode_stream(self, output_ids, input_len):
        new_ids = output_ids[0].tolist()[input_len:]
        words = []
        for idx in new_ids:
            if idx in self.id_to_word:
                word = self.id_to_word[idx]
                words.append(word)
        return words, new_ids

def load_system(root_dir):
    print(f"{YELLOW}[SYSTEM] Initializing 42D Hyper Console...{RESET}")
    
    vocab_path = root_dir / "study_session_logs" / "vocab.txt"
    checkpoint_path = root_dir / "study_session_logs" / "model_42d_latest.pt"
    
    # Load Vocab
    gen = SyntheticDataGenerator()
    if not vocab_path.exists():
        print(f"{RED}❌ Vocabulary not found!{RESET}")
        return None, None
        
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word, idx = parts
                gen.vocab[word] = int(idx)
    gen.vocab_size = len(gen.vocab)
    
    # Load Model
    if not checkpoint_path.exists():
        print(f"{RED}❌ Checkpoint not found at {checkpoint_path}{RESET}")
        return None, None
        
    print(f"{MAGENTA}>> Loading 42D Hyper Model...{RESET}")
    try:
        # Initialize Model
        config = HyperConfig(vocab_size=gen.vocab_size, d_model=256, n_layers=6, n_heads=8)
        model = HyperCosmicTransformer(config)
        
        # Load Weights
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print(f"{MAGENTA}>> 42D System Online.{RESET}")
        return gen, model
        
    except Exception as e:
        print(f"{RED}❌ Failed to load model: {e}{RESET}")
        return None, None

def print_header():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{MAGENTA}{'='*60}")
    print(f"🌌 42D HYPER CONSOLE (RESEARCH)")
    print(f"   {RESET}Commands: /temp [0.1-2.0], /reset, /exit")
    print(f"{MAGENTA}{'='*60}{RESET}\n")

# Synesthetic Color Map (Simulating 42D Engine Choice)
def get_synesthetic_color(word, prev_hash):
    # The "Engine" chooses the color based on the semantic "vibe" of the word
    # We simulate this by hashing the word + context
    h = hash(word + str(prev_hash)) % 6
    colors = [
        "\033[91m", # Red (Hot)
        "\033[93m", # Yellow (Warm)
        "\033[92m", # Green (Natural)
        "\033[96m", # Cyan (Logical)
        "\033[94m", # Blue (Deep)
        "\033[95m", # Magenta (Creative)
    ]
    return colors[h], h

def chat_loop():
    gen, model = load_system(ROOT_DIR)
    if not gen or not model: return
    
    cm = ConversationManager(gen.vocab)
    print_header()
    
    # Thinking Mode State
    thinking = False
    
    while True:
        try:
            user_input = input(f"{GREEN}👤 YOU: {RESET}")
            
            # Commands
            if user_input.startswith("/"):
                cmd = user_input.split()
                if cmd[0] in ["/exit", "/quit"]: break
                elif cmd[0] == "/reset":
                    cm.history_ids = []
                    print(f"{YELLOW}>> Memory wiped.{RESET}")
                    continue
                elif cmd[0] == "/temp":
                    try:
                        cm.temperature = float(cmd[1])
                        print(f"{YELLOW}>> Temperature set to {cm.temperature}{RESET}")
                    except: print(f"{RED}>> Invalid value.{RESET}")
                    continue
            
            # Chat
            cm.add_user_input(user_input)
            context = cm.get_context()
            
            print(f"{MAGENTA}🌌 42D: {RESET}", end="", flush=True)
            
            with torch.no_grad():
                # Generate
                output_ids = model.generate(
                    context, 
                    max_new_tokens=100, # Increased for thinking
                    temperature=cm.temperature
                )
                
                # Decode & Stream
                words, new_token_ids = cm.decode_stream(output_ids, context.size(1))
                
                prev_hash = 0
                for word in words:
                    # Thinking Mode Logic
                    if "<think>" in word:
                        thinking = True
                        print(f"\n{'\033[90m'}   (Thinking: ", end="", flush=True) # Dark Gray
                        continue
                    if "</think>" in word:
                        thinking = False
                        print(f"){RESET}\n{MAGENTA}🌌 42D: {RESET}", end="", flush=True)
                        continue
                        
                    if thinking:
                        print(word + " ", end="", flush=True)
                    else:
                        # Synesthetic Rendering
                        color, prev_hash = get_synesthetic_color(word, prev_hash)
                        print(f"{color}{word}{RESET} ", end="", flush=True)
                        
                    time.sleep(0.03)
                
                print() # Newline
                
                # Update memory with model's response
                cm.add_model_output(new_token_ids)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n{RED}❌ Error: {e}{RESET}")

if __name__ == "__main__":
    chat_loop()
