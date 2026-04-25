import sys
import os
from pathlib import Path
import torch

# Setup paths
ROOT_DIR = Path(os.getcwd())
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator

def test_inference():
    print("Initializing inference test...")
    gen = SyntheticDataGenerator()
    vocab_path = Path("study_session_logs/vocab.txt")
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2: gen.vocab[parts[0]] = int(parts[1])
        gen.vocab_size = len(gen.vocab)
    
    config = CosmicConfig(
        vocab_size=gen.vocab_size,
        d_model=256, 
        n_layers=6,
        n_heads=8
    )
    model = CosmicSynapseTransformer(config)
    
    polished_path = Path("study_session_logs/polished_12d_brain.pt")
    if polished_path.exists():
        print(f"Loading weights from {polished_path}")
        model.load_state_dict(torch.load(polished_path, map_location='cpu'))
    
    model.eval()
    
    prompt = "<|human|>\nHello, Cosmic Davis!\n<|assistant|>\n"
    token_list = []
    for word in prompt.split():
        token_list.append(gen.vocab.get(word, gen.vocab.get("<UNK>", 1)))
    
    input_ids = torch.tensor([token_list], dtype=torch.long)
    
    print("Generating...")
    with torch.no_grad():
        output_ids, _ = model.generate(input_ids, max_new_tokens=20)
    
    id_to_word = {v: k for k, v in gen.vocab.items()}
    output_tokens = output_ids[0].tolist()
    output_text = " ".join([id_to_word.get(idx, "<UNK>") for idx in output_tokens])
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output_text}")
    
    if len(output_text.split()) > len(prompt.split()):
        print("\n[SUCCESS] Inference produced tokens.")
    else:
        print("\n[FAILED] Inference failed to produce tokens.")

if __name__ == "__main__":
    test_inference()
