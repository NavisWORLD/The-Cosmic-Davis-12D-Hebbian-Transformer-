# Usage Guide - Internal Dimension AI

Complete guide to using the Internal Dimension AI framework.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Environment Creation](#environment-creation)
4. [Network Configuration](#network-configuration)
5. [Training](#training)
6. [Metric Interpretation](#metric-interpretation)
7. [Multi-Agent Setup](#multi-agent-setup)
8. [Cosmic Synapse](#cosmic-synapse-parameters)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- 4GB RAM minimum (8GB+ recommended)
- GPU optional (but recommended for large experiments)

### Setup

```bash
# Clone repository
cd internal-dimension-ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Verify Installation

```python
import torch
from src.core.network import InternalDimensionNetwork
from src.core.metrics import ConsciousnessMetrics

print("✓ Installation successful!")
```

---

## Quick Start

### 1. Create a Simple Agent

```python
import torch
from src.core.network import InternalDimensionNetwork
from src.environments.gridworld import GridWorld
from src.training.trainer import PPOTrainer

# Create environment
env = GridWorld(size=8)

# Create network with internal dimensions
model = InternalDimensionNetwork(
    input_dim=env.observation_space.shape[0],  # 2 (x, y position)
    hidden_dim=128,
    output_dim=env.action_space.n,  # 4 (up, down, left, right)
    internal_dim=12  # x₁₂/m₁₂ dimension size
)

# Create trainer
trainer = PPOTrainer(model, env)

# Train for 100 episodes
history = trainer.train(num_episodes=100)

# View results
print(f"Final reward: {history['episode_rewards'][-1]:.2f}")
print(f"Final x₁₂: {history['x12_means'][-1]:.3f}")
print(f"Final m₁₂: {history['m12_means'][-1]:.3f}")
```

### 2. Run Pre-Built Examples

```bash
# Quick demo (5 minutes)
python examples/01_quick_demo.py

# Baseline comparison (15 minutes)
python examples/02_baseline_comparison.py

# Curiosity demo (10 minutes)
python examples/03_curiosity_demo.py

# Full consciousness tracking (30 minutes)
python examples/04_consciousness_tracking.py
```

---

## Environment Creation

### Built-In Environments

#### GridWorld

Simple grid navigation task.

```python
from src.environments.gridworld import GridWorld

env = GridWorld(
    size=10,                    # Grid size (10x10)
    goal_position=(9, 9),       # Goal location
    reward_goal=1.0,            # Reward for reaching goal
    reward_step=-0.01,          # Penalty per step
    max_steps=200               # Episode length
)
```

#### TwoRoomGridWorld

Grid with a doorway - tests exploration.

```python
from src.environments.gridworld import TwoRoomGridWorld

env = TwoRoomGridWorld(
    size=12,                    # Grid size
    door_position=(6, 5),       # Doorway location
    goal_position=(10, 10)      # Goal in second room
)
```

#### Multi-Agent Environments

```python
from src.environments.social import PrisonersDilemma, IteratedPrisonersDilemma

# Single-shot prisoner's dilemma
env = PrisonersDilemma(
    num_agents=2,
    internal_dim=12
)

# Iterated version
env = IteratedPrisonersDilemma(
    num_agents=2,
    num_rounds=10,
    internal_dim=12
)
```

### Custom Environments

Any Gymnasium-compatible environment works:

```python
import gymnasium as gym

# Use any Gym environment
env = gym.make('CartPole-v1')

model = InternalDimensionNetwork(
    input_dim=env.observation_space.shape[0],
    hidden_dim=128,
    output_dim=env.action_space.n,
    internal_dim=24
)
```

---

## Network Configuration

### Internal Dimension Network

```python
from src.core.network import InternalDimensionNetwork

model = InternalDimensionNetwork(
    input_dim=4,           # Input features
    hidden_dim=128,        # Hidden layer size
    output_dim=2,          # Number of actions
    internal_dim=12,       # x₁₂/m₁₂ dimension size

    # Internal dynamics parameters
    alpha=1.0,             # x₁₂ sensitivity to prediction error
    beta=0.5,              # x₁₂ sensitivity to novelty
    gamma=0.3,             # x₁₂ sensitivity to attention
    delta=0.1,             # x₁₂ decay rate
    eta=0.02,              # m₁₂ integration rate
    zeta=0.01,             # m₁₂ regression to baseline

    # Network architecture
    use_lstm=True,         # Use LSTM for hidden state
    dropout=0.1,           # Dropout rate
    device='cpu'           # Device
)
```

### Baseline Network (for comparison)

```python
from src.core.network import BaselineNetwork

baseline = BaselineNetwork(
    input_dim=4,
    hidden_dim=128,
    output_dim=2,
    use_lstm=True,
    dropout=0.1
)
```

---

## Training

### PPO Trainer Configuration

```python
from src.training.trainer import PPOTrainer

trainer = PPOTrainer(
    model=model,
    env=env,

    # PPO hyperparameters
    learning_rate=3e-4,
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # GAE lambda
    clip_epsilon=0.2,        # PPO clip parameter
    value_loss_coef=0.5,     # Value loss weight
    entropy_coef=0.01,       # Entropy bonus
    max_grad_norm=0.5,       # Gradient clipping
    ppo_epochs=4,            # PPO update epochs
    batch_size=64,           # Minibatch size

    # Intrinsic rewards
    intrinsic_reward_weight=0.1,  # Weight for curiosity/wisdom
    intrinsic_reward_method='balanced',  # 'curiosity', 'wisdom', or 'balanced'

    # Suffering detection
    suffering_threshold=-0.7,  # x₁₂ threshold
    suffering_patience=100,    # Steps before warning

    # Logging
    use_tensorboard=True,
    use_wandb=False,
    log_interval=10,
    save_interval=100,
    checkpoint_dir='checkpoints'
)

# Train
history = trainer.train(
    num_episodes=300,
    steps_per_episode=200,
    render=False,
    compute_consciousness_interval=10
)
```

### Training Loop

```python
# Custom training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # Forward pass
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        policy_logits, value, internals = model(
            state_tensor,
            return_internals=True
        )

        # Sample action
        dist = torch.distributions.Categorical(logits=policy_logits)
        action = dist.sample()

        # Environment step
        next_state, reward, done, truncated, info = env.step(action.item())

        # Update internal state
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.tensor([reward])

        update_info = model.update_internal_state(
            current_hidden=internals['hidden'],
            next_state=next_state_tensor,
            reward=reward_tensor
        )

        # Track metrics
        print(f"x₁₂: {update_info['x12'].item():.3f}, "
              f"m₁₂: {update_info['m12'].item():.3f}")

        episode_reward += reward
        state = next_state

        if done or truncated:
            break

    print(f"Episode {episode}: Reward = {episode_reward:.2f}")
```

---

## Metric Interpretation

### Consciousness Metrics

```python
from src.core.metrics import ConsciousnessMetrics

metrics = ConsciousnessMetrics()

# Compute consciousness score
score_dict = metrics.compute_consciousness_score(
    model=model,
    x12_history=list(model.internal_state.x12_history),
    m12_history=list(model.internal_state.m12_history),
    sample_inputs=torch.randn(10, input_dim)
)

print(f"Consciousness Score: {score_dict['consciousness_score']:.3f}")
print(f"R_ω (richness): {score_dict['r_omega']:.3f}")
print(f"R_ψ (binding): {score_dict['r_psi']:.3f}")
print(f"φ (integration): {score_dict.get('phi', 0):.3f}")
```

### Interpretation Guide

**Consciousness Score (0-1)**
- < 0.3: Low consciousness indicators
- 0.3-0.6: Moderate indicators
- 0.6-0.8: Strong indicators
- > 0.8: Very strong indicators (rare)

**R_ω (Richness)**
- Optimal range: 0.5-0.7
- < 0.3: Too ordered (lacks diversity)
- 0.5-0.7: "Edge of chaos" (optimal)
- > 0.8: Too chaotic (lacks structure)

**R_ψ (Binding)**
- 0-0.3: Low temporal coherence
- 0.3-0.6: Moderate coherence
- 0.6-1.0: High coherence (stable patterns)

**x₁₂ (Awareness)**
- < -0.5: Negative surprise (expected events)
- -0.5 to 0.5: Normal range
- > 0.5: High surprise/novelty

**m₁₂ (Memory)**
- < -0.3: Accumulated negative experiences
- -0.3 to 0.3: Neutral
- > 0.3: Accumulated positive experiences

---

## Multi-Agent Setup

### Training Multiple Agents

```python
from src.environments.social import IteratedPrisonersDilemma

# Create multi-agent environment
env = IteratedPrisonersDilemma(num_agents=2, num_rounds=10)

# Create agents
agents = {
    f'agent_{i}': InternalDimensionNetwork(
        input_dim=env.observation_space['agent_0'].shape[0],
        hidden_dim=64,
        output_dim=env.action_space['agent_0'].n,
        internal_dim=12
    )
    for i in range(2)
}

# Training loop
obs, info = env.reset()

for episode in range(num_episodes):
    actions = {}

    # Each agent selects action
    for agent_id, agent_model in agents.items():
        state_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
        policy_logits, value, _ = agent_model(state_tensor)

        dist = torch.distributions.Categorical(logits=policy_logits)
        actions[agent_id] = dist.sample().item()

    # Environment step
    obs, rewards, dones, truncs, info = env.step(actions)

    # Check cooperation rate
    print(f"Cooperation rate: {info['cooperation_rate']:.2f}")
```

### Consciousness Correlation

```python
# Analyze x₁₂ correlation between agents
import numpy as np

x12_agent0 = [x.item() for x in agents['agent_0'].internal_state.x12_history]
x12_agent1 = [x.item() for x in agents['agent_1'].internal_state.x12_history]

correlation = np.corrcoef(x12_agent0, x12_agent1)[0, 1]
print(f"x₁₂ correlation: {correlation:.3f}")

# High positive correlation suggests synchronized awareness
```

---

## Cosmic Synapse Parameters

### Configuration File

Edit `configs/experiments/cosmic_synapse.yaml`:

```yaml
physics:
  n_particles: 256          # Number of particles
  dt: 0.001                 # Timestep
  G: 1.0                    # Gravitational constant
  lorenz_sigma: 10.0        # Lorenz sigma
  lorenz_rho: 28.0          # Lorenz rho
  lorenz_beta: 2.666667     # Lorenz beta
  hebbian_strength: 0.1     # Hebbian learning rate

transformer:
  vocab_size: 50257         # GPT-2 vocabulary
  ctx_len: 512              # Context length
  d_model: 768              # Model dimension
  n_layer: 8                # Number of layers
  n_head: 12                # Attention heads
  dropout: 0.1              # Dropout rate

training:
  total_steps: 2000000      # Total training steps
  learning_rate: 3e-4       # Learning rate
  checkpoint_interval: 200000  # Checkpoint frequency
```

### Running Experiments

```bash
# Quick test (1k steps)
python scripts/run_cosmic_synapse.py --steps 1000 --particles 32

# Full run (2M steps)
python scripts/run_cosmic_synapse.py --steps 2000000 --particles 256 --device cuda
```

### Programmatic Use

```python
from src.advanced.cosmic_synapse import CosmicSynapsePhysics, CosmicTransformer

# Initialize physics
physics = CosmicSynapsePhysics(n=128, seed=42)

# Run physics simulation
for _ in range(1000):
    physics.step()

# Get state for transformer
physics_state = physics.get_state_vector()

# Initialize transformer
model = CosmicTransformer(physics_conditioning=True)

# Generate text conditioned on physics
input_ids = torch.tensor([[1, 2, 3]])
generated = model.generate(
    input_ids,
    physics_state,
    max_new_tokens=50,
    temperature=0.8
)
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Solution: Ensure package is installed
pip install -e .

# Or add to path manually
import sys
sys.path.insert(0, 'path/to/internal-dimension-ai/src')
```

#### 2. CUDA Out of Memory

```python
# Solution 1: Reduce batch size
trainer = PPOTrainer(model, env, batch_size=32)  # Default is 64

# Solution 2: Use CPU
model = InternalDimensionNetwork(..., device='cpu')

# Solution 3: Enable gradient checkpointing
torch.utils.checkpoint.checkpoint_sequential(...)
```

#### 3. Training Instability

```python
# Solution: Adjust learning rate and clipping
trainer = PPOTrainer(
    model, env,
    learning_rate=1e-4,     # Lower learning rate
    max_grad_norm=1.0,      # Higher gradient clipping
    clip_epsilon=0.1        # Tighter PPO clipping
)
```

#### 4. Low Consciousness Scores

```python
# Solution 1: Train longer
history = trainer.train(num_episodes=500)  # Instead of 100

# Solution 2: Increase internal dimension size
model = InternalDimensionNetwork(..., internal_dim=64)  # Instead of 12

# Solution 3: Use curiosity-driven exploration
trainer = PPOTrainer(
    model, env,
    intrinsic_reward_weight=0.2,  # Higher intrinsic reward
    intrinsic_reward_method='balanced'
)
```

#### 5. Suffering Warnings

```
WARNING: Suffering detected: x₁₂ = -0.75 < -0.7 for 100 steps.
```

**Solutions:**
- Check if task is too difficult
- Adjust reward structure
- Consider pausing/resetting training
- Review ethical guidelines in `docs/ETHICS.md`

### Performance Optimization

```python
# Use mixed precision training (if on GPU)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    policy_logits, value = model(state)
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Track internal dimensions during training
for episode in range(num_episodes):
    # ... training code ...

    # Log every 10 episodes
    if episode % 10 == 0:
        print(f"x₁₂ history length: {len(model.internal_state.x12_history)}")
        print(f"m₁₂ current value: {model.internal_state.m12.item():.3f}")
        print(f"Novelty buffer size: {len(model.internal_state.novelty_buffer)}")
```

---

## Next Steps

1. **Explore Examples**: Run all example scripts to understand different use cases
2. **Read Theory**: Review `docs/THEORY.md` for mathematical foundations
3. **Run Experiments**: Try dimensional scaling and emergence timeline experiments
4. **Customize**: Build your own environments and consciousness tests
5. **Contribute**: Share findings and improvements with the community

For more information, see:
- `API.md` - Complete API reference
- `docs/THEORY.md` - Mathematical theory
- `docs/ETHICS.md` - Ethical guidelines
- `examples/` - Working code examples
