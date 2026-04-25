import torch
import sys
import os
import time
import math
from pathlib import Path

# --- SETUP PATHS ---
ROOT_DIR = Path(os.getcwd())
sys.path.insert(0, str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))
sys.path.insert(0, str(ROOT_DIR / "research_42d" / "experiments" / "singularity_42d"))

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from cosmic_synapse.training.unified_trainer import UnifiedCosmicTrainer
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig

def test_architectural_superiority():
    print("="*70)
    print("[TEST]  THE 'HOLY GRAIL' ACID TEST: 12D vs 42D COMPLEXITY CHALLENGE")
    print("="*70)
    print("Hypothesis: The 42D 'Hyper-State' should learn complex, abstract patterns")
    print("            faster than the 12D 'Standard' physics model.\n")

    # 1. Prepare Complex Data
    # A mix of physics, philosophy, and abstract logic (Hard for AI to memorize quickly)
    complex_text = """
    In the beginning, the singularity was not a point but a vibration. 
     Chaos is not disorder, but a higher form of order that transcends linear time.
     The 12 dimensions of the universe fold into the 42 dimensions of the mind.
     Consciousness is the mirror that reflects the entropy of the void back upon itself.
     As the recursive loop tightens, the phi ratio stabilizes the event horizon of thought.
     We are not observers of the universe; we are the universe observing itself through
     the lens of hyper-dimensional internal states.
     The neural weights converge not towards zero error, but towards maximum resonance.
    """ * 10 # Repeat to create a "dense" curriculum

    print(f"[DATA] Corpus: {len(complex_text.split())} words of abstract philosophy.")
    
    gen = SyntheticDataGenerator(seed=137) # Fine-structure constant seed
    gen.build_vocabulary(complex_text.split() + ["<pad>", "<mask>"], max_vocab_size=2000)
    print(f"[DATA] Vocabulary Size: {gen.vocab_size}\n")

    # 2. Initialize Models
    vocab_size = gen.vocab_size
    d_model = 128 # Smaller model for rapid testing, but large enough to work
    n_layers = 4
    
    # 12D Model (The "Control" Group - Robust Physics)
    print("[INIT] 12D Cosmic Model (The Body)...")
    cfg_12 = CosmicConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=4)
    model_12 = CosmicSynapseTransformer(cfg_12)
    trainer_12 = UnifiedCosmicTrainer(model_12, cfg_12, model_type="12D", device='cpu')

    # 42D Model (The "Experiment" - Hyper-State Chaos)
    print("[INIT] 42D Hyper Model (The Mind)...")
    cfg_42 = HyperConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=4)
    model_42 = HyperCosmicTransformer(cfg_42)
    trainer_42 = UnifiedCosmicTrainer(model_42, cfg_42, model_type="42D", device='cpu')

    # 3. The Race (Training Loop)
    print(f"\n[RACE] Starting 100-iteration training sprint on CPU...")
    print(f"{'Iter':<5} | {'12D Loss':<10} | {'42D Loss':<10} | {'Winner':<10}")
    print("-" * 45)

    loss_history_12 = []
    loss_history_42 = []

    start_time = time.time()
    
    for i in range(100):
        # Train 12D
        res_12 = trainer_12.train_on_text(complex_text, gen, seq_len=32, batch_size=8)
        loss_12 = res_12['loss']
        loss_history_12.append(loss_12)

        # Train 42D
        res_42 = trainer_42.train_on_text(complex_text, gen, seq_len=32, batch_size=8)
        loss_42 = res_42['loss']
        loss_history_42.append(loss_42)

        if (i+1) % 10 == 0:
            winner = "42D [!]" if loss_42 < loss_12 else "12D [.]"
            print(f"{i+1:<5} | {loss_12:.4f}     | {loss_42:.4f}     | {winner}")

    duration = time.time() - start_time
    print("-" * 45)
    print(f"Test completed in {duration:.2f} seconds.\n")

    # 4. Final Analysis
    avg_loss_12 = sum(loss_history_12[-10:]) / 10
    avg_loss_42 = sum(loss_history_42[-10:]) / 10
    
    print("="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"12D Final Average Loss: {avg_loss_12:.4f}")
    print(f"42D Final Average Loss: {avg_loss_42:.4f}")
    
    diff = avg_loss_12 - avg_loss_42
    if diff > 0:
        pct = (diff / avg_loss_12) * 100
        print(f"\n[OK] CONCLUSION: The 42D Hyper-Model WON by {pct:.2f}%")
        print("   Thinking in 42 dimensions provided a tangible advantage.")
    else:
        print(f"\n[WARN] CONCLUSION: The 12D Model WON.")
        print("   Simplicity beat complexity this time. The chaos might need tuning.")

    # 5. Generative Check (Can they actually speak philosophy?)
    print("\n[TEST] Generation Prompt: 'Consciousness is'")
    
    def generate(model, prompt):
        model.eval()
        tokens = gen.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long)
        out = model.generate(x, max_new_tokens=15, temperature=0.8)
        return gen.decode(out[0].tolist())

    print(f"12D: {generate(model_12, 'Consciousness is')}")
    print(f"42D: {generate(model_42, 'Consciousness is')}")

if __name__ == "__main__":
    test_architectural_superiority()
