# Installation Guide - 12D Cosmic Synapse Transformer

Complete installation instructions for all platforms.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Installation](#quick-installation)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Troubleshooting](#troubleshooting)
- [GPU Setup](#gpu-setup)
- [Development Installation](#development-installation)

## Prerequisites

### Required
- **Python**: 3.8 or higher
- **pip**: Latest version recommended
- **Git**: For cloning the repository

### Recommended
- **CUDA**: 11.8+ for GPU support (NVIDIA GPUs)
- **16GB RAM**: For training medium models
- **Virtual environment**: conda, venv, or virtualenv

## Quick Installation

### For Users (Stable Release)

Once published to PyPI, install with:

```bash
pip install cosmic-synapse-transformer
```

### For Developers (Latest Code)

```bash
# Clone the repository
git clone https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine.git

# Navigate to the project
cd "infinite-adaptive-audio-12d-universe-engine/internal ai test"

# Install in editable mode
pip install -e .
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Update system packages
sudo apt-get update
sudo apt-get install -y python3 python3-pip git

# Clone repository
git clone https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine.git
cd "infinite-adaptive-audio-12d-universe-engine/internal ai test"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "from cosmic_synapse_transformer import CosmicSynapseTransformer; print('✓ Installation successful!')"
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10 git

# Clone repository
git clone https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine.git
cd "infinite-adaptive-audio-12d-universe-engine/internal ai test"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "from cosmic_synapse_transformer import CosmicSynapseTransformer; print('✓ Installation successful!')"
```

### Windows

```powershell
# Install Python from python.org or Microsoft Store
# Install Git from git-scm.com

# Clone repository
git clone https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine.git
cd "infinite-adaptive-audio-12d-universe-engine\internal ai test"

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "from cosmic_synapse_transformer import CosmicSynapseTransformer; print('✓ Installation successful!')"
```

## GPU Setup

### NVIDIA CUDA

For GPU acceleration, install CUDA-enabled PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Apple Silicon (M1/M2/M3)

PyTorch has MPS (Metal Performance Shaders) support:

```bash
# Install PyTorch with MPS support
pip install torch torchvision

# Verify MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Use MPS in configs
# training:
#   device: 'mps'
```

## Development Installation

For contributing to the project:

```bash
# Clone repository
git clone https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine.git
cd "infinite-adaptive-audio-12d-universe-engine/internal ai test"

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify
pytest tests/ -v
```

## Docker Installation

### Using Docker

```bash
# Build image
docker build -t cosmic-transformer .

# Run container
docker run -it cosmic-transformer bash

# Or use docker-compose
docker-compose up cosmic-dev
```

### Using Docker with GPU

```bash
# Install nvidia-docker
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Run with GPU
docker run --gpus all -it cosmic-transformer bash
```

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'cosmic_synapse_transformer'"

**Solution**: Install the package in editable mode
```bash
pip install -e .
```

#### "ImportError: cannot import name 'CosmicConfig'"

**Solution**: Ensure you're in the correct directory and have installed dependencies
```bash
cd "internal ai test"
pip install -r requirements.txt
```

#### "RuntimeError: CUDA out of memory"

**Solutions**:
1. Reduce batch size in config file
2. Use gradient accumulation
3. Enable gradient checkpointing
4. Use a smaller model

```yaml
training:
  batch_size: 2  # Reduce this
  gradient_accumulation_steps: 8  # Increase this
```

#### "Slow training on CPU"

**Solutions**:
1. Use GPU if available
2. Reduce model size (use tiny_model.yaml)
3. Enable PyTorch compilation (requires PyTorch 2.0+)

```yaml
training:
  device: 'cuda'  # or 'mps' for Apple Silicon
  compile: true
```

#### Windows: "error: Microsoft Visual C++ 14.0 or greater is required"

**Solution**: Install Microsoft C++ Build Tools
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Install "Desktop development with C++"

#### macOS: "clang: error: unsupported option '-fopenmp'"

**Solution**: Install OpenMP
```bash
brew install libomp
```

### Verifying Installation

Run this comprehensive verification script:

```python
import sys
import torch
from cosmic_synapse_transformer import (
    CosmicConfig,
    CosmicSynapseTransformer
)

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

# Create tiny model
config = CosmicConfig(
    vocab_size=100,
    max_seq_len=64,
    d_model=96,
    n_layers=2,
    n_heads=2,
)

model = CosmicSynapseTransformer(config)
print(f"✓ Model created with {model.get_num_params():,} parameters")

# Test forward pass
inputs = torch.randint(0, 100, (1, 32))
outputs = model(inputs)
print(f"✓ Forward pass successful, output shape: {outputs.shape}")

print("\n✅ All checks passed! Installation is complete.")
```

## Updating

### Update to Latest Version

```bash
# From PyPI (when available)
pip install --upgrade cosmic-synapse-transformer

# From source
cd "infinite-adaptive-audio-12d-universe-engine/internal ai test"
git pull
pip install -e . --upgrade
```

## Uninstallation

```bash
pip uninstall cosmic-synapse-transformer
```

## Next Steps

After installation:

1. ✅ Run quick start: `bash scripts/quick_start.sh`
2. ✅ Read the [Quick Start Guide](QUICKSTART.md)
3. ✅ Try examples in `examples/`
4. ✅ Train your first model

## Getting Help

- 📖 [Documentation](https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine)
- 🐛 [Report Issues](https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine/issues)
- 💬 [Discussions](https://github.com/NavisWORLD/infinite-adaptive-audio-12d-universe-engine/discussions)
- 📧 Email: cory@cosmicsynapse.ai
