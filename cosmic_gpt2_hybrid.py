"""
COSMIC DAVIS - GPT-2 WEIGHT LOADER
===================================
Loads pre-trained GPT-2 weights into the Cosmic Davis 12D architecture.

This gives you:
- Instant language capability from GPT-2
- PLUS the 12D Hebbian/Chaos innovations on top
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional

# Setup paths
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(ROOT_DIR / "app" / "modules" / "dual_mind"))

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig


class CosmicGPT2Hybrid(nn.Module):
    """
    Cosmic Davis model initialized with GPT-2 weights.
    
    Standard transformer components come from GPT-2.
    12D innovations (Hebbian attention, internal state, chaos) are added on top.
    """
    
    def __init__(self, gpt2_path: str = "pretrained_weights/gpt2-medium"):
        super().__init__()
        
        print("="*60)
        print("🌌 COSMIC DAVIS + GPT-2 HYBRID MODEL")
        print("="*60)
        
        # Load GPT-2
        print("\n[1/4] Loading GPT-2 weights...")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
        
        # Add pad token (GPT-2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        gpt2_config = self.gpt2.config
        print(f"   ✅ GPT-2 loaded: {gpt2_config.n_embd}d, {gpt2_config.n_layer} layers, {gpt2_config.n_head} heads")
        
        # Create matching Cosmic config
        print("\n[2/4] Creating Cosmic Davis config...")
        self.config = CosmicConfig(
            vocab_size=gpt2_config.vocab_size,
            max_seq_len=gpt2_config.n_positions,
            d_model=gpt2_config.n_embd,
            n_layers=gpt2_config.n_layer,
            n_heads=gpt2_config.n_head,
        )
        # Override phi-optimization to match GPT-2 exactly
        self.config.d_model = gpt2_config.n_embd
        self.config.d_ff = gpt2_config.n_embd * 4
        self.config.d_k = gpt2_config.n_embd // gpt2_config.n_head
        
        # Create Cosmic model
        print("\n[3/4] Initializing Cosmic Davis 12D architecture...")
        self.cosmic = CosmicSynapseTransformer(self.config)
        
        # Transfer weights
        print("\n[4/4] Transferring GPT-2 weights to Cosmic Davis...")
        self._transfer_weights()
        
        print("\n" + "="*60)
        print("✅ HYBRID MODEL READY")
        print("   - Language capability: GPT-2 (pre-trained)")
        print("   - 12D Enhancements: Cosmic Davis (trainable)")
        print("="*60)
    
    def _transfer_weights(self):
        """Copy GPT-2 weights to matching Cosmic components."""
        gpt2 = self.gpt2.transformer
        cosmic = self.cosmic
        
        # 1. Embeddings
        cosmic.token_embedding.weight.data.copy_(gpt2.wte.weight.data)
        cosmic.position_embedding.weight.data.copy_(gpt2.wpe.weight.data)
        print("   ✅ Embeddings transferred")
        
        # 2. Each layer
        for i, (cosmic_layer, gpt2_layer) in enumerate(zip(cosmic.layers, gpt2.h)):
            # Layer norms
            cosmic_layer.ln1.weight.data.copy_(gpt2_layer.ln_1.weight.data)
            cosmic_layer.ln1.bias.data.copy_(gpt2_layer.ln_1.bias.data)
            cosmic_layer.ln2.weight.data.copy_(gpt2_layer.ln_2.weight.data)
            cosmic_layer.ln2.bias.data.copy_(gpt2_layer.ln_2.bias.data)
            
            # Attention - GPT-2 uses fused QKV, we need to split
            qkv_weight = gpt2_layer.attn.c_attn.weight.data  # [d_model, 3*d_model]
            qkv_bias = gpt2_layer.attn.c_attn.bias.data
            d = self.config.d_model
            
            # GPT-2 uses Conv1D which is transposed
            cosmic_layer.attention.W_Q.weight.data.copy_(qkv_weight[:, :d].t())
            cosmic_layer.attention.W_K.weight.data.copy_(qkv_weight[:, d:2*d].t())
            cosmic_layer.attention.W_V.weight.data.copy_(qkv_weight[:, 2*d:].t())
            
            if cosmic_layer.attention.W_Q.bias is not None:
                cosmic_layer.attention.W_Q.bias.data.copy_(qkv_bias[:d])
                cosmic_layer.attention.W_K.bias.data.copy_(qkv_bias[d:2*d])
                cosmic_layer.attention.W_V.bias.data.copy_(qkv_bias[2*d:])
            
            # Output projection
            cosmic_layer.attention.W_O.weight.data.copy_(gpt2_layer.attn.c_proj.weight.data.t())
            if cosmic_layer.attention.W_O.bias is not None:
                cosmic_layer.attention.W_O.bias.data.copy_(gpt2_layer.attn.c_proj.bias.data)
            
            # Feed-forward
            cosmic_layer.ffn.W1.weight.data.copy_(gpt2_layer.mlp.c_fc.weight.data.t())
            cosmic_layer.ffn.W1.bias.data.copy_(gpt2_layer.mlp.c_fc.bias.data)
            cosmic_layer.ffn.W2.weight.data.copy_(gpt2_layer.mlp.c_proj.weight.data.t())
            cosmic_layer.ffn.W2.bias.data.copy_(gpt2_layer.mlp.c_proj.bias.data)
        
        print(f"   ✅ {len(cosmic.layers)} layers transferred")
        
        # 3. Final layer norm
        cosmic.ln_f.weight.data.copy_(gpt2.ln_f.weight.data)
        cosmic.ln_f.bias.data.copy_(gpt2.ln_f.bias.data)
        print("   ✅ Final layer norm transferred")
        
        # Note: lm_head is tied to token_embedding in our model, so it's already copied
        
    def forward(self, input_ids, targets=None):
        """Forward pass using the Cosmic architecture with GPT-2 knowledge."""
        return self.cosmic(input_ids, targets=targets)
    
    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7, top_k: int = 50) -> str:
        """Generate text from prompt."""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Generate using Cosmic model
        self.cosmic.eval()
        with torch.no_grad():
            output_ids = self.cosmic.generate(
                input_ids, 
                max_new_tokens=max_tokens, 
                temperature=temperature,
                top_k=top_k
            )
        
        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
    
    def chat(self, user_input: str, max_tokens: int = 100) -> str:
        """Simple chat interface."""
        # Format as conversation
        prompt = f"User: {user_input}\nAssistant:"
        response = self.generate(prompt, max_tokens=max_tokens, temperature=0.7)
        
        # Extract assistant response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response


def test_hybrid_model():
    """Test the hybrid model."""
    print("\n" + "="*60)
    print("🧪 TESTING COSMIC-GPT2 HYBRID")
    print("="*60)
    
    # Load hybrid
    hybrid = CosmicGPT2Hybrid()
    
    # Test prompts
    test_prompts = [
        "Hello, my name is",
        "The meaning of life is",
        "Artificial intelligence will",
        "In the future, humans",
    ]
    
    print("\n📝 Generation Tests:")
    for prompt in test_prompts:
        output = hybrid.generate(prompt, max_tokens=20, temperature=0.8)
        print(f"\n   Prompt: '{prompt}'")
        print(f"   Output: '{output}'")
    
    # Interactive chat
    print("\n" + "="*60)
    print("💬 INTERACTIVE CHAT (type 'quit' to exit)")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 🌌")
                break
            
            response = hybrid.chat(user_input)
            print(f"\n🤖 Cosmic Davis: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 🌌")
            break


if __name__ == "__main__":
    test_hybrid_model()
