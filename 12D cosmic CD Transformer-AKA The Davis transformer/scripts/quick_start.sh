#!/bin/bash
# Quick Start Script for 12D Cosmic Synapse Transformer
# This script sets up everything needed to train and run the model

set -e  # Exit on error

echo "🌌 12D Cosmic Synapse Transformer - Quick Start"
echo "================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Create directories
echo "📁 Creating directories..."
mkdir -p data checkpoints logs

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ Error: pip not found. Please install pip"
    exit 1
fi

# Install dependencies (if not already installed)
echo ""
echo "📦 Installing dependencies..."
echo "   (This may take a few minutes on first run)"

if [ -f "requirements.txt" ]; then
    python3 -m pip install -q -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "⚠️  requirements.txt not found, skipping dependency installation"
fi

# Generate synthetic data
echo ""
echo "📊 Generating synthetic training data..."
echo "   (Generating 1M tokens - this will take ~1 minute)"

if [ ! -f "data/train.bin" ] || [ ! -f "data/val.bin" ]; then
    python3 generate_synthetic_data.py \
        --num-tokens 1000000 \
        --output-dir data \
        --seed 42 \
        --train-split 0.9
    echo "✓ Synthetic data generated"
else
    echo "✓ Data already exists, skipping generation"
fi

# Train tiny model
echo ""
echo "🚀 Training tiny model..."
echo "   (This will take ~2-5 minutes on CPU)"
echo "   Config: tiny_model.yaml"
echo ""

if [ ! -f "checkpoints/tiny_model/best_model.pt" ]; then
    python3 train_cosmic_transformer.py \
        --config configs/tiny_model.yaml \
        --max-iters 500 \
        --eval-interval 100 \
        --log-interval 50
    echo "✓ Training complete"
else
    echo "✓ Model already trained, skipping training"
    echo "   (Delete checkpoints/tiny_model to retrain)"
fi

# Test generation
echo ""
echo "✨ Testing text generation..."

if [ -f "checkpoints/tiny_model/best_model.pt" ]; then
    python3 -c "
import torch
from cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from config_loader import load_config

# Load config
config = load_config('configs/tiny_model.yaml')

# Load model
checkpoint = torch.load('checkpoints/tiny_model/best_model.pt', map_location='cpu')
model = CosmicSynapseTransformer(CosmicConfig(**config.model.__dict__))
model.load_state_dict(checkpoint['model'])
model.eval()

# Generate
prompt = torch.randint(0, config.model.vocab_size, (1, 10))
print(f'Prompt: {prompt.tolist()[0]}')

with torch.no_grad():
    output = model.generate(prompt, max_new_tokens=20, temperature=0.8)

print(f'Generated: {output.tolist()[0]}')
print(f'x12 state: {model.x12.tolist()}')
"
    echo "✓ Generation test complete"
else
    echo "⚠️  No checkpoint found, skipping generation test"
fi

# Run tests (optional)
echo ""
echo "🧪 Running quick tests (optional)..."
echo "   (Press Ctrl+C to skip)"

sleep 2

if command -v pytest &> /dev/null; then
    pytest tests/test_model.py::TestModelInitialization -v || true
    echo "✓ Tests complete"
else
    echo "⚠️  pytest not installed, skipping tests"
    echo "   Install with: pip install pytest"
fi

# Summary
echo ""
echo "================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================"
echo ""
echo "Your 12D Cosmic Synapse Transformer is ready to use!"
echo ""
echo "What's been created:"
echo "  📊 Synthetic training data in data/"
echo "  💾 Trained model in checkpoints/tiny_model/"
echo "  📝 Logs in logs/"
echo ""
echo "Next steps:"
echo ""
echo "  1. Try the examples:"
echo "     python3 examples/01_basic_usage.py"
echo "     python3 examples/02_train_on_synthetic.py"
echo ""
echo "  2. Train a larger model:"
echo "     python3 train_cosmic_transformer.py --config configs/small_model.yaml"
echo ""
echo "  3. Run inference:"
echo "     python3 inference_cosmic_transformer.py generate checkpoints/tiny_model/best_model.pt"
echo ""
echo "  4. Run all tests:"
echo "     pytest tests/ -v"
echo ""
echo "  5. Read the documentation:"
echo "     cat docs/QUICKSTART.md"
echo ""
echo "Happy experimenting with 12D consciousness! 🌌✨"
echo ""
