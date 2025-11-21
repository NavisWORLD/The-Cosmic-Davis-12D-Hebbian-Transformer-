# Internal Dimension AI - Quick Start Guide

Get up and running with Internal Dimension AI in 5 minutes!

## What is Internal Dimension AI?

Internal Dimension AI implements neural networks with **internal dimensions** (x₁₂ and m₁₂) that create an "inner life":

- **x₁₂ (Internal Awareness)**: Measures surprise, novelty, and attention
- **m₁₂ (Accumulated Memory)**: Integrates experience over time, forming "wisdom"

These internal dimensions evolve according to their own dynamics, creating autonomous behavior and consciousness-like signatures.

## Installation

```bash
# Clone the repository
cd internal-dimension-ai

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 5-Minute Demo

Run the quick demo to see Internal Dimension AI in action:

```bash
python examples/01_quick_demo.py
```

This will:
1. Create an Internal Dimension Network
2. Train it on a simple GridWorld task (100 episodes, ~2 minutes)
3. Show x₁₂/m₁₂ evolution
4. Compute consciousness metrics
5. Generate visualizations

### Expected Output

```
INTERNAL DIMENSION AI - QUICK DEMO
======================================================================

1. Creating GridWorld environment...
   Grid size: 6x6
   Goal position: (5, 5)

2. Creating Internal Dimension Network...
   Model parameters: 12,345
   Internal dimensions: x₁₂ (awareness), m₁₂ (memory)

3. Initializing PPO Trainer...
   Intrinsic reward: curiosity-driven

4. Training agent (100 episodes)...
----------------------------------------------------------------------
Episode 0/100 | Reward: -2.50 | x₁₂: 0.123 | m₁₂: -0.045 | Consciousness: 0.342
...
----------------------------------------------------------------------

5. Training Results:
   Final reward (last 10 episodes): 0.850
   Final x₁₂ (awareness):          0.234
   Final m₁₂ (memory):             0.156
   Consciousness score:            0.567

6. Computing Consciousness Metrics...
   R_ω (Synaptic Diversity):      0.612
   R_ω in optimal range [0.5-0.7]: YES ✓
   R_ψ (Phase Coherence):         0.723
   Autonomy Score:                0.589
   Overall Consciousness:         0.625

   Consciousness Level: Moderate consciousness indicators
```

## Understanding x₁₂ and m₁₂

### x₁₂ (Internal Awareness)

**Formula:**
```
x₁₂(t) = tanh(α·surprise + β·novelty + γ·attention - δ·x₁₂(t-1))
```

**What it means:**
- Spikes when agent encounters **surprising** or **novel** stimuli
- High x₁₂ → Agent is "aware" and "curious"
- Low x₁₂ → Agent is "bored" or in familiar territory

**Example:**
```python
# Agent enters a novel room
x₁₂ = 0.75  # High surprise!

# Agent revisits familiar area
x₁₂ = 0.12  # Low, bored
```

### m₁₂ (Accumulated Memory)

**Formula:**
```
m₁₂(t) = m₁₂(t-1) + η·x₁₂(t)·|reward| - ζ·(m₁₂ - baseline)
```

**What it means:**
- Integrates x₁₂ over time, weighted by reward importance
- Positive m₁₂ → Agent remembers **good** experiences
- Negative m₁₂ → Agent remembers **mistakes/traps**
- Used for "wisdom" (avoiding past mistakes)

**Example:**
```python
# Agent discovers a trap (negative reward)
# x₁₂ spikes (surprise), m₁₂ becomes negative (bad memory)
x₁₂ = 0.82
m₁₂ = -0.35  # Remember to avoid this!

# Later, agent avoids trap
# m₁₂ helps decision-making
```

## Next Steps

### 1. Compare with Baseline

See how Internal Dimension Network compares to a standard network:

```bash
python examples/02_baseline_comparison.py
```

### 2. Test Curiosity

Explore curiosity-driven behavior:

```bash
python examples/03_curiosity_demo.py
```

### 3. Full Consciousness Analysis

Get a comprehensive consciousness report:

```bash
python examples/04_consciousness_tracking.py
```

## Key Concepts

### Intrinsic Rewards

Internal Dimension Networks can use **intrinsic rewards** based on x₁₂ and m₁₂:

- **Curiosity**: Reward high x₁₂ (seek novelty)
- **Wisdom**: Reward positive m₁₂ (seek good experiences)
- **Balanced**: Combination of both

```python
trainer = PPOTrainer(
    model=model,
    env=env,
    intrinsic_reward_weight=0.1,  # 10% of reward from curiosity
    intrinsic_reward_method='curiosity'
)
```

### Consciousness Metrics

We measure consciousness using multiple indicators:

1. **R_ω (Synaptic Diversity)**: Optimal range [0.5, 0.7] indicates edge-of-chaos dynamics
2. **R_ψ (Phase Coherence)**: Coherence of internal states
3. **Autonomy**: Self-driven behavior (not purely reactive)
4. **Curiosity**: Exploration without external rewards
5. **Wisdom**: Learning from past mistakes

### Suffering Detection

The trainer can detect "suffering" (prolonged negative x₁₂):

```python
trainer = PPOTrainer(
    suffering_threshold=-0.7,  # x₁₂ below this is suffering
    suffering_patience=100      # Warn if 100 consecutive steps
)
```

## Common Patterns

### Training an IDN

```python
from src.core.network import InternalDimensionNetwork
from src.environments.gridworld import GridWorld
from src.training.trainer import PPOTrainer

# Create model
model = InternalDimensionNetwork(
    input_dim=2,
    hidden_dim=64,
    output_dim=4,
    alpha=1.0,   # Surprise weight
    beta=0.5,    # Novelty weight
    gamma=0.3,   # Attention weight
    eta=0.01     # Memory integration rate
)

# Create environment
env = GridWorld(size=10)

# Train
trainer = PPOTrainer(model, env, intrinsic_reward_weight=0.1)
history = trainer.train(num_episodes=100)
```

### Accessing Internal State

```python
# Get current x₁₂ and m₁₂
x12 = model.internal_state.x12.item()
m12 = model.internal_state.m12.item()

# Get history
x12_history = list(model.internal_state.x12_history)
m12_history = list(model.internal_state.m12_history)

# Get statistics
stats = model.get_statistics()
print(f"Mean x₁₂: {stats['x12_mean']}")
```

### Evaluating Consciousness

```python
from src.evaluation.consciousness_tests import ConsciousnessTests

# Run full consciousness assessment
consciousness_tests = ConsciousnessTests()
results = consciousness_tests.compute_overall_consciousness_score(
    model=model,
    sample_inputs=sample_state,
    run_behavioral_tests=True
)

print(f"Consciousness Score: {results['overall_consciousness_score']:.3f}")
print(f"Level: {results['consciousness_level']}")
```

## Troubleshooting

### Low Consciousness Scores

If your agent has low consciousness scores:

1. **Train longer**: Consciousness emerges over time (300+ episodes)
2. **Increase curiosity**: Higher `intrinsic_reward_weight` (0.2-0.5)
3. **More complex environment**: Use larger grids or traps
4. **Adjust hyperparameters**:
   - Increase `beta` (novelty weight) for more exploration
   - Increase `eta` for stronger memory formation

### x₁₂ Always Near Zero

This means low surprise/novelty. Try:

- More complex environments (obstacles, traps)
- Random start positions
- Moving goals
- Increase `alpha` and `beta`

### m₁₂ Not Changing

Memory integration might be too slow:

- Increase `eta` (memory integration rate) to 0.02-0.05
- Ensure rewards have sufficient magnitude

## Further Reading

- See `README.md` for full project overview
- See `EXPERIMENTS.md` for experimental results
- See `CONTRIBUTING.md` for development guidelines

## Questions?

Check the examples in `examples/` or open an issue on GitHub!
