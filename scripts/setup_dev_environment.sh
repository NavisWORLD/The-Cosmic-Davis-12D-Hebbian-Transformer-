#!/bin/bash
# Setup development environment

echo "Setting up development environment..."

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e packages/cosmic-synapse-transformer
pip install -e packages/internal-dimension-ai

echo "Done!"
