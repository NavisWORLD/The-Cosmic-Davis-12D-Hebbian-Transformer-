"""
12D COSMIC WEB INTERFACE
========================
Host your model and share it with the world.
Uses Gradio to create a public link.
"""

import sys
import torch
import time
import os
from pathlib import Path

# Add paths
ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))

from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig

try:
    import gradio as gr
except ImportError:
    print("Please install gradio: pip install gradio")
    sys.exit(1)

# Global State
MODEL = None
GEN = None
ID_TO_WORD = {}

def load_system():
    global MODEL, GEN, ID_TO_WORD
    
    vocab_path = ROOT_DIR / "study_session_logs" / "vocab.txt"
    checkpoint_path = ROOT_DIR / "study_session_logs" / "model_12d_latest.pt"
    
    if not vocab_path.exists() or not checkpoint_path.exists():
        return "System not ready (missing vocab or checkpoint)"

    # Load Vocab
    gen = SyntheticDataGenerator()
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word, idx = parts
                gen.vocab[word] = int(idx)
    gen.vocab_size = len(gen.vocab)
    
    # Load Model
    config = CosmicConfig(vocab_size=gen.vocab_size, d_model=256, n_layers=6, n_heads=8)
    model = CosmicSynapseTransformer(config)
    
    # Load weights (map to CPU to avoid stealing GPU from training)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        return f"Error loading model: {e}"
    model.eval()
    
    MODEL = model
    GEN = gen
    ID_TO_WORD = {v: k for k, v in gen.vocab.items()}
    
    return "System Loaded Successfully"

def predict(message, history):
    if MODEL is None:
        load_system()
        
    # Encode
    input_ids = []
    for word in message.lower().split():
        if word in GEN.vocab:
            input_ids.append(GEN.vocab[word])
    
    if not input_ids:
        return "I don't know those words yet."
        
    context = torch.tensor([input_ids], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        # Temperature 0.7 for balance
        output_ids = MODEL.generate(context, max_new_tokens=100, temperature=0.7)
        
    # Decode
    new_ids = output_ids[0].tolist()[len(input_ids):]
    words = []
    
    # Formatting for Web
    response_text = ""
    thinking = False
    
    for idx in new_ids:
        if idx in ID_TO_WORD:
            word = ID_TO_WORD[idx]
            
            # Handle Thinking Tags for Markdown
            if "<think>" in word:
                thinking = True
                response_text += "**(Thinking:** "
                continue
            if "</think>" in word:
                thinking = False
                response_text += "**)**\n\n"
                continue
                
            if thinking:
                response_text += f"*{word}* "
            else:
                response_text += f"{word} "
                
    return response_text

# Initialize
print("Loading 12D Model for Web...")
load_system()

# UI Definition
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌌 12D Cosmic Synapse")
    gr.Markdown("### The First Digital Mind | Production Interface")
    
    chat = gr.ChatInterface(
        predict,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Talk to the 12D Model...", container=False, scale=7),
        title=None,
        description="This model is currently training. Responses may be abstract.",
        theme="soft",
        examples=["What is 1+1?", "Define Chaos", "Who are you?"],
        cache_examples=False,
    )
    
    with gr.Accordion("System Controls", open=False):
        reload_btn = gr.Button("Reload Latest Brain Checkpoint")
        reload_btn.click(load_system, outputs=[])

if __name__ == "__main__":
    # share=True creates the public link
    demo.launch(share=True)
