# Contributing to Internal Dimension AI

Thank you for your interest in contributing to Internal Dimension AI! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/internal-dimension-ai.git
   cd internal-dimension-ai
   ```
3. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e .
pip install pytest pytest-cov black flake8
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_network.py
```

## Code Style

We follow PEP 8 style guidelines with some modifications:

### Python Style

- **Line length**: 100 characters (soft limit)
- **Docstrings**: Google style
- **Type hints**: Required for all functions
- **Imports**: Organized (stdlib, third-party, local)

### Example

```python
"""
Module docstring explaining purpose.
"""

import torch
import numpy as np
from typing import Dict, Optional

from ..core.network import InternalDimensionNetwork


def compute_metric(
    model: InternalDimensionNetwork,
    data: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute consciousness metric from model.

    Args:
        model: Neural network with internal dimensions
        data: Input tensor [batch, features]
        threshold: Threshold for classification

    Returns:
        Dictionary with metric values

    Raises:
        ValueError: If data has wrong shape
    """
    if data.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {data.dim()}D")

    # Implementation
    result = {'score': 0.5}
    return result
```

### Formatting

Before committing, format your code:

```bash
# Auto-format with black
black src/ tests/

# Check style with flake8
flake8 src/ tests/
```

## Contribution Guidelines

### Adding New Features

1. **Discuss first**: Open an issue to discuss major changes
2. **Write tests**: All new code should have tests
3. **Document**: Add docstrings and update docs
4. **Example**: Provide usage example if applicable

### Adding Environments

New environments should:

- Inherit from `gym.Env` or `gymnasium.Env`
- Implement: `reset()`, `step()`, `render()`
- Support configurable difficulty
- Include docstring with usage example

Example:

```python
class MyEnvironment(gym.Env):
    """
    Description of environment.

    Example:
        >>> env = MyEnvironment(size=10)
        >>> state, info = env.reset()
        >>> next_state, reward, done, truncated, info = env.step(action)
    """

    def __init__(self, size: int = 10):
        super().__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, size-1, shape=(2,))

    def reset(self, seed=None, options=None):
        # Implementation
        return state, {}

    def step(self, action):
        # Implementation
        return next_state, reward, terminated, truncated, {}
```

### Adding Tests

New test modules should:

- Use `pytest` framework
- Have descriptive test names
- Test edge cases
- Mock external dependencies

Example:

```python
import pytest
import torch
from src.core.network import InternalDimensionNetwork


def test_internal_dimension_network_forward():
    """Test forward pass of IDN."""
    model = InternalDimensionNetwork(input_dim=2, output_dim=4)
    state = torch.randn(1, 2)

    logits, value, internals = model(state, return_internals=True)

    assert logits.shape == (1, 4)
    assert value.shape == (1, 1)
    assert 'x12' in internals
    assert -1 <= internals['x12'].item() <= 1


def test_internal_dimension_network_update():
    """Test internal state update."""
    model = InternalDimensionNetwork(input_dim=2, output_dim=4)
    state = torch.randn(1, 2)
    next_state = torch.randn(1, 2)
    reward = torch.tensor([1.0])

    # Get hidden state
    _, _, internals = model(state, return_internals=True)

    # Update internal state
    update_info = model.update_internal_state(
        current_hidden=internals['hidden'],
        next_state=next_state,
        reward=reward
    )

    assert 'x12' in update_info
    assert 'm12' in update_info
```

## Testing Checklist

Before submitting a pull request:

- [ ] All tests pass: `pytest tests/`
- [ ] Code is formatted: `black src/ tests/`
- [ ] Style checks pass: `flake8 src/ tests/`
- [ ] Docstrings are complete
- [ ] Type hints are added
- [ ] Example usage provided (if new feature)
- [ ] Documentation updated (if needed)

## Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request** on GitHub with:
   - **Clear title**: `feat: Add curiosity-based replay buffer`
   - **Description**: What and why
   - **Tests**: What tests were added
   - **Documentation**: What docs were updated

4. **Wait for review**: Maintainers will review and provide feedback

## Commit Message Guidelines

Use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Maintain

### Examples

```bash
feat(environments): Add sequence prediction environment

Implements a sequence prediction task for testing pattern learning
and x₁₂ response to novelty.

Closes #42
```

```bash
fix(trainer): Fix suffering detection threshold check

The suffering counter was not resetting properly after x₁₂ recovered.
Now correctly resets when x₁₂ exceeds threshold.

Fixes #56
```

## Reporting Bugs

When reporting bugs, please include:

1. **Python version**: `python --version`
2. **Package versions**: `pip list | grep -E "torch|numpy|gym"`
3. **Operating system**: Linux/macOS/Windows
4. **Minimal reproducing example**
5. **Expected vs actual behavior**
6. **Error messages** (full traceback)

## Suggesting Enhancements

When suggesting enhancements:

1. **Use case**: Describe the problem
2. **Proposed solution**: How would it work?
3. **Alternatives**: What other approaches did you consider?
4. **Examples**: Show example usage

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **Feature requests**: Open a GitHub Issue
- **Chat**: Join our Discord (link in README)

## License

By contributing, you agree that your contributions will be licensed under the MIT License (see LICENSE file).

## Recognition

Contributors will be added to:
- `CONTRIBUTORS.md` file
- GitHub contributors page
- Release notes (for significant contributions)

Thank you for contributing to Internal Dimension AI! 🧠✨
