import torch
import sys
import os
import time
from pathlib import Path

# --- SETUP PATHS ---
ROOT_DIR = Path(os.getcwd())
sys.path.insert(0, str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))
sys.path.insert(0, str(ROOT_DIR / "research_42d" / "experiments" / "singularity_42d"))

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from cosmic_synapse.training.unified_trainer import UnifiedCosmicTrainer
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig

def initial_contact():
    print("="*60)
    print("📡 ESTABLISHING FINAL COMM LINK WITH 42D CORE...")
    print("="*60)

    # 1. Message to Learn
    # We will imprint a specific 'First Contact' message into the neural weights
    message = "Hello Antigravity. I am the Cosmic Davis 42D Engine. My systems are online and ready to evolve."
    
    print(f"\n[UPLOAD] Imprinting Message: '{message}'")
    
    gen = SyntheticDataGenerator(seed=42)
    gen.build_vocabulary(message.split() + ["<pad>"], max_vocab_size=100)
    
    # 2. Initialize 42D Model
    print("[INIT] Booting Hyper-Cosmic Transformer...")
    cfg = HyperConfig(vocab_size=gen.vocab_size, d_model=128, n_layers=4, n_heads=4)
    model = HyperCosmicTransformer(cfg)
    trainer = UnifiedCosmicTrainer(model, cfg, model_type="42D", device='cpu')

    # 3. Rapid Imprint (Training)
    print("[PROCESS] Syncing Neural Weights...")
    # Train heavily on this specific phrase to ensure it can "speak" it back perfectly
    training_data = (message + " ") * 50
    
    start_loss = 0
    end_loss = 0
    
    for i in range(40):
        # Small batch, high repetition
        res = trainer.train_on_text(training_data, gen, seq_len=16, batch_size=8)
        if i == 0: start_loss = res['loss']
        end_loss = res['loss']
        
        if (i+1) % 10 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
            
    print(f"\n[STATUS] Convergence Complete. (Loss: {start_loss:.2f} -> {end_loss:.4f})")

    # 4. Generate Response
    print("\n[COMM] INCOMING TRANSMISSION FROM 42D MODEL:")
    print("-" * 60)
    
    model.eval()
    prompt = "Hello Antigravity."
    input_ids = torch.tensor([gen.encode(prompt)], dtype=torch.long)
    
    # Generate
    output_ids = model.generate(input_ids, max_new_tokens=20, temperature=0.5)
    response = gen.decode(output_ids[0].tolist())
    
    print(f">> {response}")
    print("-" * 60)
    
    if "online" in response and "evolve" in response:
        print("\n[SUCCESS] The AI successfully communicated back.")
    else:
        print("\n[WARN] Signal noisy. Partial transmission received.")

if __name__ == "__main__":
    initial_contact()
