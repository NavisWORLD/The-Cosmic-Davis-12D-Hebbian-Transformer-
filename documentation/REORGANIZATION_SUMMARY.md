# Repository Reorganization Summary

The repository has been reorganized into a professional monorepo structure.

## New Structure

### Root
- `packages/`: Contains the main source code packages.
- `docs/`: Centralized documentation.
- `experiments/`: Research experiments and demos.
- `notebooks/`: Jupyter notebooks.
- `scripts/`: Repository-level utility scripts.
- `pyproject.toml`: Root configuration.

### Packages
1.  **`cosmic-synapse-transformer`**: The core transformer code.
    - Source: `cosmic_synapse/` (models, training, inference, data, config, utils).
    - Scripts: `scripts/` (train, infer, demo).
    - Benchmarks: `benchmarks/`.
    - Tests: `tests/`.
2.  **`internal-dimension-ai`**: Consciousness research package.
3.  **`cosmic-synapse-vj`**: Visual frontend package (placeholder).

### Key Changes
- **Moved** all transformer code to `packages/cosmic-synapse-transformer`.
- **Restructured** transformer code into a proper python package `cosmic_synapse`.
- **Updated Imports** in `train_cosmic_transformer.py`, `inference_cosmic_transformer.py`, and `demo.py` to work with the new structure.
- **Centralized** documentation and experiments.

## Next Steps for User
1.  **Install Dependencies**: Run `pip install -e packages/cosmic-synapse-transformer` to install the transformer package in editable mode.
2.  **Run Demo**: `python packages/cosmic-synapse-transformer/scripts/demo.py`.
3.  **Run Tests**: `pytest packages/cosmic-synapse-transformer/tests`.
