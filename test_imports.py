import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))

print("Testing imports...")
try:
    import torch
    print("torch ok")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("transformers ok")
    from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
    print("cosmic_synapse ok")
    print("All imports successful!")
except Exception as e:
    print(f"Import failed: {e}")
