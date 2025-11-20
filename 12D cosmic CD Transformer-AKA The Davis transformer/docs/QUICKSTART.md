# 🚀 Quick Start Guide - 12D Cosmic Synapse Transformer

Get up and running with the 12D Cosmic Synapse Transformer in under 5 minutes!

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA for GPU support

### Step 1: Clone the Repository

```bash
git clone https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine.git
cd "infinite-adaptive-audio-12d-universe-engine/internal ai test"
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or for editable install:

```bash
pip install -e .
```

### Step 3: Run Quick Start Script

```bash
bash scripts/quick_start.sh
```

This will:
1. ✅ Create necessary directories
2. ✅ Generate synthetic training data (1M tokens)
3. ✅ Train a tiny model (~5 minutes on CPU)
4. ✅ Test text generation
5. ✅ Run quick tests

## Manual Setup (Alternative)

If you prefer to set up manually:

### 1. Generate Synthetic Data

```bash
python generate_synthetic_data.py \
    --num-tokens 1000000 \
    --output-dir data \
    --seed 42
```

### 2. Train a Model

```bash
python train_cosmic_transformer.py \
    --config configs/tiny_model.yaml
```

### 3. Generate Text

```bash
python inference_cosmic_transformer.py generate \
    checkpoints/tiny_model/best_model.pt \
    --prompt "Hello world" \
    --max-tokens 50
```

## Your First Training Run

### Using Python Code

```python
from cosmic_synapse_transformer import CosmicConfig, CosmicSynapseTransformer
import torch

# Create a tiny model
config = CosmicConfig(
    vocab_size=1000,
    max_seq_len=128,
    d_model=192,
    n_layers=4,
    n_heads=4,
)

model = CosmicSynapseTransformer(config)
print(f"Parameters: {model.get_num_params():,}")

# Create dummy data
inputs = torch.randint(0, 1000, (2, 128))
targets = inputs.clone()

# Train one step
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
optimizer.zero_grad()
logits, loss = model(inputs, targets)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
print(f"x12 state: {model.x12}")
```

### Using Configuration Files

```bash
# Tiny model (CPU, ~5 min)
python train_cosmic_transformer.py --config configs/tiny_model.yaml

# Small model (GPU, ~1-2 hours)
python train_cosmic_transformer.py --config configs/small_model.yaml

# Medium model (V100, ~12-24 hours)
python train_cosmic_transformer.py --config configs/medium_model.yaml
```

## Running Examples

```bash
# Basic usage
python examples/01_basic_usage.py

# Full training on synthetic data
python examples/02_train_on_synthetic.py

# Interactive demo with visualization
python examples/03_interactive_demo.py
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_model.py -v

# Quick tests only
pytest tests/ -v -m "not slow"
```

## Docker Deployment

```bash
# Build image
docker build -t cosmic-transformer .

# Run API server
docker run -p 5000:5000 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    cosmic-transformer

# Or use docker-compose
docker-compose up cosmic-transformer
```

## API Usage

Start the API server:

```bash
python inference_cosmic_transformer.py serve \
    checkpoints/tiny_model/best_model.pt \
    --host 0.0.0.0 \
    --port 5000
```

Make requests:

```python
import requests

response = requests.post('http://localhost:5000/generate', json={
    'prompt': 'The cosmic synapse',
    'max_tokens': 50,
    'temperature': 0.8
})

print(response.json()['completion'])
```

## Common Issues

### Issue: Out of Memory

**Solution**: Use a smaller model or reduce batch size

```yaml
# In your config file
training:
  batch_size: 2  # Reduce this
  gradient_accumulation_steps: 4  # Increase this
```

### Issue: Slow Training

**Solution**: Enable GPU or model compilation

```yaml
training:
  device: 'cuda'  # Use GPU
  compile: true   # PyTorch 2.0 compilation
```

### Issue: Module not found

**Solution**: Install in editable mode

```bash
pip install -e .
```

## Next Steps

1. **Read the Theory**: [docs/THEORY.md](THEORY.md)
2. **Explore Examples**: Check the `examples/` directory
3. **Train Larger Models**: Try `configs/small_model.yaml`
4. **Visualize Results**: Use `utils/visualization.py`
5. **Contribute**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

## Getting Help

- 📖 **Documentation**: [docs/](.)
- 🐛 **Issues**: [GitHub Issues](https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine/discussions)
- 📧 **Email**: cory@cosmicsynapse.ai

## Quick Reference

| Command | Purpose |
|---------|---------|
| `bash scripts/quick_start.sh` | One-command setup |
| `python generate_synthetic_data.py` | Create training data |
| `python train_cosmic_transformer.py --config <yaml>` | Train model |
| `python inference_cosmic_transformer.py generate <model>` | Generate text |
| `pytest tests/ -v` | Run tests |
| `docker-compose up` | Run in Docker |

Happy experimenting with 12D consciousness! 🌌✨
