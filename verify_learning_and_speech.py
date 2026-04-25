import torch
import sys
import os
import time
from pathlib import Path

# --- SETUP PATHS ---
ROOT_DIR = Path(os.getcwd())
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(ROOT_DIR / "research_42d" / "experiments" / "singularity_42d"))

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from cosmic_synapse.training.unified_trainer import UnifiedCosmicTrainer
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig

def test_learning_capability():
    print("="*60)
    print("VERIFYING LEARNING & MEMORY (12D vs 42D)")
    print("="*60)

    # 1. Prepare Data with a specific "Fact" to learn
    # We repeat a pattern so learning is obvious in a short time.
    fact = "The secret password is Omega."
    print(f"\n[GOAL] Teach the models this fact: '{fact}'")
    
    gen = SyntheticDataGenerator(seed=42)
    # Build vocab including these words
    gen.build_vocabulary(fact.split() + ["Blue", "Red", "Apple"], max_vocab_size=100)
    
    # Create training data (Repeat the fact 20 times)
    training_text = (fact + " ") * 20
    print(f"[DATA] Corpus length: {len(training_text.split())} words")

    # 2. Initialize Models
    print("\n[INIT] Initializing Neural Engines...")
    vocab_size = gen.vocab_size
    
    # 12D Config
    cfg_12 = CosmicConfig(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=2)
    model_12 = CosmicSynapseTransformer(cfg_12)
    trainer_12 = UnifiedCosmicTrainer(model_12, cfg_12, model_type="12D", device='cpu')

    # 42D Config
    cfg_42 = HyperConfig(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=2)
    model_42 = HyperCosmicTransformer(cfg_42)
    trainer_42 = UnifiedCosmicTrainer(model_42, cfg_42, model_type="42D", device='cpu')

    # 3. Training Loop
    print("\n[TRAINING] Running rapid learning cycle (50 steps)...")
    
    start_loss_12 = 0
    start_loss_42 = 0
    end_loss_12 = 0
    end_loss_42 = 0

    for i in range(50):
        # We retrain on the same text to force memorization (overfitting test)
        res_12 = trainer_12.train_on_text(training_text, gen, seq_len=8, batch_size=4)
        res_42 = trainer_42.train_on_text(training_text, gen, seq_len=8, batch_size=4)
        
        if i == 0:
            start_loss_12 = res_12['loss']
            start_loss_42 = res_42['loss']
            print(f"  Step 1  | 12D Loss: {start_loss_12:.4f} | 42D Loss: {start_loss_42:.4f}")
        
        if (i+1) % 10 == 0:
             print(f"  Step {i+1} | 12D Loss: {res_12['loss']:.4f} | 42D Loss: {res_42['loss']:.4f}")

    end_loss_12 = res_12['loss']
    end_loss_42 = res_42['loss']

    # 4. Analyze Results
    print("\n[RESULTS] Learning Efficiency:")
    drop_12 = start_loss_12 - end_loss_12
    drop_42 = start_loss_42 - end_loss_42
    
    print(f"  12D Model: Loss dropped by {drop_12:.4f}")
    if end_loss_12 < 1.0:
        print("  [OK] 12D has successfully memorized the pattern.")
    else:
        print("  [WARN] 12D is struggling to memorize.")

    print(f"  42D Model: Loss dropped by {drop_42:.4f}")
    if end_loss_42 < 1.0:
        print("  [OK] 42D has successfully memorized the pattern.")
    else:
        print("  [WARN] 42D is struggling to memorize.")

    # 5. Verify Speaking/Recall
    print("\n[TEST] Text Generation (Recall):")
    prompt = "The secret password"
    print(f"  Prompt: '{prompt}'")
    
    # Function to generate text
    def generate(model, input_str):
        model.eval()
        tokens = gen.encode(input_str)
        if not tokens: return "[Error: Encoding failed]"
        
        x = torch.tensor([tokens], dtype=torch.long)
        # Generate 3 tokens (should be "is", "Omega", ".")
        out_ids = model.generate(x, max_new_tokens=3)
        return gen.decode(out_ids[0].tolist())

    out_12 = generate(model_12, prompt)
    print(f"  12D Says: '{out_12}'")
    
    out_42 = generate(model_42, prompt)
    print(f"  42D Says: '{out_42}'")

    if "Omega" in out_12 and "Omega" in out_42:
        print("\nSUCCESS: Both models learned and remembered the secret fact!")
    elif "Omega" in out_12:
        print("\nPARTIAL: Only 12D remembered.")
    elif "Omega" in out_42:
        print("\nPARTIAL: Only 42D remembered.")
    else:
        print("\nFAIL: Neither model produced the correct word.")

if __name__ == "__main__":
    test_learning_capability()
