import torch
import sys
import math
import time
from pathlib import Path
import numpy as np

# Add paths
ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator

def load_latest_model():
    """Find and load the most recent 12D checkpoint."""
    log_dir = ROOT_DIR / "study_session_logs"
    vocab_path = log_dir / "vocab.txt"
    checkpoint_dir = log_dir / "checkpoints"
    
    if not vocab_path.exists():
        print("[ERROR] No vocabulary found. Has training started?")
        return None, None

    # Load vocab
    gen = SyntheticDataGenerator()
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                gen.vocab[parts[0]] = int(parts[1])
    gen.vocab_size = len(gen.vocab)
    
    # Find latest checkpoint
    checkpoint_path = None
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("model_12d_iter_*.pt"))
        if checkpoints:
            checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
            checkpoint_path = checkpoints[-1]
    
    if not checkpoint_path:
        legacy_path = log_dir / "model_12d_latest.pt"
        if legacy_path.exists():
            checkpoint_path = legacy_path
        else:
            print("[ERROR] No checkpoints found.")
            return None, None
            
    print(f"[OK] Loading checkpoint: {checkpoint_path.name}")
    
    # Load model
    config = CosmicConfig(vocab_size=gen.vocab_size, d_model=256, n_layers=6, n_heads=8)
    model = CosmicSynapseTransformer(config)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model, gen
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None, None

def evaluate_perplexity(model, gen, text_sample):
    """Calculate perplexity on a text sample."""
    print("\n[BENCHMARK] Evaluating Perplexity...")
    
    # Tokenize
    input_ids = []
    for word in text_sample.split():
        if word in gen.vocab:
            input_ids.append(gen.vocab[word])
        else:
            # Skip unknown words or map to unk if available
            pass
            
    if len(input_ids) < 2:
        print("[ERROR] Text sample too short or vocab mismatch.")
        return
        
    # Create batches
    seq_len = 64
    total_loss = 0
    num_batches = 0
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i in range(0, len(input_ids) - seq_len, seq_len):
            x = torch.tensor([input_ids[i:i+seq_len]], dtype=torch.long).to(device)
            y = torch.tensor([input_ids[i+1:i+seq_len+1]], dtype=torch.long).to(device)
            
            _, loss, _ = model(x, y)
            total_loss += loss.item()
            num_batches += 1
            
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        print(f"   - Average Loss: {avg_loss:.4f}")
        print(f"   - Perplexity:   {perplexity:.2f}")
        
        if perplexity < 100:
            print("   [Analysis] Excellent! Model is predicting with high confidence.")
        elif perplexity < 1000:
            print("   [Analysis] Good. Model has learned basic structure.")
        else:
            print("   [Analysis] High perplexity. Model is still in early learning phase.")
    else:
        print("[WARN] Not enough data for a full batch.")

if __name__ == "__main__":
    model, gen = load_latest_model()
    
    if model:
        # Standard benchmark text (Alice in Wonderland snippet)
        benchmark_text = """
        Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?'
        So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.
        There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, 'Oh dear! Oh dear! I shall be too late!' (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.
        """
        
        evaluate_perplexity(model, gen, benchmark_text)
