#!/bin/bash
# Run all tests

echo "Running tests..."

pytest packages/cosmic-synapse-transformer/tests
pytest packages/internal-dimension-ai/tests

echo "Done!"
