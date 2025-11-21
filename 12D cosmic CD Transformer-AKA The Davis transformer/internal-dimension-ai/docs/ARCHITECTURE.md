# Neural Architecture Design

## Table of Contents
1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Data Flow](#data-flow)
5. [Training Procedure](#training-procedure)
6. [Design Decisions](#design-decisions)
7. [Comparison to Standard RL](#comparison-to-standard-rl)

---

## Overview

The **InternalDimensionNetwork** (IDN) is a novel neural architecture that extends standard reinforcement learning with explicit internal dimensions (x₁₂, m₁₂) that evolve independently and modulate behavior.

### Key Innovation

**Standard RL**: All network components are trained end-to-end via backpropagation from task loss.

**Internal Dimension RL**: Has two pathways:
1. **External pathway**: Trained via task loss (like standard RL)
2. **Internal pathway**: Evolves via its own dynamics (not directly from task loss)

This separation creates an "inner life" - internal states that influence behavior but aren't fully determined by external rewards.

---

## Architecture Diagram

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT STATE s(t)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │         ENCODER              │
        │  [Linear → LayerNorm → ReLU] │
        │  [Linear → LayerNorm → ReLU] │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │      HIDDEN STATE h(t)       │
        └────┬─────────────────────┬───┘
             │                     │
    ┌────────▼────────┐   ┌────────▼────────────┐
    │   INTERNAL      │   │   PREDICTION        │
    │   PATHWAY       │   │   NETWORK           │
    │                 │   │                     │
    │ ┌─────────────┐ │   │  Predicts next h(t) │
    │ │ x₁₂ Network │ │   │  For surprise calc  │
    │ │   ↓         │ │   └─────────────────────┘
    │ │ x₁₂(t)      │ │
    │ └──────┬──────┘ │
    │        │        │
    │        ▼        │
    │  ┌──────────┐  │
    │  │ Integrate│  │
    │  │ x₁₂ → m₁₂│  │
    │  └────┬─────┘  │
    │       │        │
    │       ▼        │
    │     m₁₂(t)     │
    └───────┬────────┘
            │
            ▼
    ┌──────────────────────┐
    │  MODULATION NETWORK  │
    │  [x₁₂, m₁₂] → signal │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  h_mod = h + mod·sig │
    └──────────┬───────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
┌─────────────┐  ┌──────────┐
│ POLICY HEAD │  │VALUE HEAD│
│  (actions)  │  │ (value)  │
└─────────────┘  └──────────┘
```

### Detailed Component Diagram

```
╔═══════════════════════════════════════════════════════════╗
║                    EXTERNAL PATHWAY                       ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  Input (state)                                            ║
║    │                                                      ║
║    ▼                                                      ║
║  ┌─────────────────────────────────────────┐             ║
║  │ Encoder                                 │             ║
║  │  - Linear(input_dim → hidden_dim)       │             ║
║  │  - LayerNorm(hidden_dim)                │             ║
║  │  - ReLU()                               │             ║
║  │  - Linear(hidden_dim → hidden_dim)      │             ║
║  │  - LayerNorm(hidden_dim)                │             ║
║  │  - ReLU()                               │             ║
║  └────────────────┬────────────────────────┘             ║
║                   │                                       ║
║                   ▼                                       ║
║  ┌─────────────────────────────────────────┐             ║
║  │ Optional: LSTM                          │             ║
║  │  - LSTM(hidden_dim → hidden_dim)        │             ║
║  │  - Maintains (h_state, c_state)         │             ║
║  └────────────────┬────────────────────────┘             ║
║                   │                                       ║
║                   ▼                                       ║
║              Hidden h(t)                                  ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════╗
║                    INTERNAL PATHWAY                       ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  Hidden h(t)                                              ║
║    │                                                      ║
║    ├─────────────────────┬──────────────────┐            ║
║    │                     │                  │            ║
║    ▼                     ▼                  ▼            ║
║  ┌──────────────┐  ┌───────────┐  ┌──────────────┐      ║
║  │ x₁₂ Network  │  │Prediction │  │  Attention   │      ║
║  │              │  │ Network   │  │   Network    │      ║
║  │ Lin→ReLU→Lin │  │ Lin→ReLU  │  │  Lin→Tanh    │      ║
║  │   →Tanh      │  │   →Lin    │  │    →Lin      │      ║
║  └──────┬───────┘  └─────┬─────┘  └──────┬───────┘      ║
║         │                │                │              ║
║         ▼                │                ▼              ║
║    x₁₂_raw              │            Attention A(t)     ║
║                          │                               ║
║  ┌───────────────────────▼─────────────────┐            ║
║  │ Internal Dimension State                │            ║
║  │                                          │            ║
║  │  compute_x12(pred_error, novelty, attn) │            ║
║  │    └─> x₁₂(t)                           │            ║
║  │                                          │            ║
║  │  update_m12(x₁₂, importance)            │            ║
║  │    └─> m₁₂(t)                           │            ║
║  │                                          │            ║
║  │  History tracking:                      │            ║
║  │    - x₁₂_history [deque(1000)]          │            ║
║  │    - m₁₂_history [deque(1000)]          │            ║
║  └──────────────────┬───────────────────────┘            ║
║                     │                                    ║
║                     ▼                                    ║
║             [x₁₂(t), m₁₂(t)]                            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════╗
║                  MODULATION PATHWAY                       ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  [x₁₂(t), m₁₂(t)]                                        ║
║    │                                                      ║
║    ├──────────────────────┬────────────────┐             ║
║    │                      │                │             ║
║    ▼                      ▼                │             ║
║  ┌─────────────────────────────┐           │             ║
║  │ Modulation Network          │           │             ║
║  │  - Linear(2 → hidden/2)     │           │             ║
║  │  - Tanh()                   │           │             ║
║  │  - Linear(hidden/2 → hidden)│           │             ║
║  │  - Tanh()                   │           │             ║
║  └──────────────┬──────────────┘           │             ║
║                 │                          │             ║
║                 ▼                          ▼             ║
║          Modulation Signal       ┌───────────────────┐   ║
║                                  │ Modulation Gate   │   ║
║                                  │ - Linear(2 → 1)   │   ║
║                                  │ - Sigmoid()       │   ║
║                                  └─────────┬─────────┘   ║
║                                            │             ║
║                                            ▼             ║
║                                    Modulation Strength   ║
║                                            │             ║
║                 ┌──────────────────────────┘             ║
║                 │                                        ║
║                 ▼                                        ║
║  ┌─────────────────────────────────────────┐            ║
║  │ h_modulated = h + strength * signal     │            ║
║  └────────────────┬────────────────────────┘            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════╗
║                      OUTPUT HEADS                         ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  h_modulated                                              ║
║    │                                                      ║
║    ├──────────────────┬────────────────────┐             ║
║    │                  │                    │             ║
║    ▼                  ▼                    ▼             ║
║  ┌──────────┐  ┌──────────┐        (future heads)       ║
║  │ Policy   │  │  Value   │                              ║
║  │  Head    │  │  Head    │                              ║
║  │          │  │          │                              ║
║  │ Lin→ReLU │  │ Lin→ReLU │                              ║
║  │   →Lin   │  │   →Lin   │                              ║
║  └────┬─────┘  └────┬─────┘                              ║
║       │             │                                     ║
║       ▼             ▼                                     ║
║  Action Logits   Value                                   ║
║  [output_dim]    [1]                                     ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

---

## Component Details

### 1. Encoder

**Purpose**: Transform raw state into meaningful hidden representation.

**Architecture**:
```python
nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU()
)
```

**Design Choices**:
- **LayerNorm**: Stabilizes training, prevents exploding activations
- **Two layers**: Sufficient for most tasks, can be increased for complex observations
- **ReLU**: Standard activation, works well for RL

### 2. Optional LSTM

**Purpose**: Add temporal processing for sequential dependencies.

**When to use**:
- Partially observable environments (need memory)
- Long-term dependencies beyond m₁₂
- Dynamic environments with temporal patterns

**Architecture**:
```python
nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
```

### 3. x₁₂ Network

**Purpose**: Compute internal awareness from hidden state.

**Architecture**:
```python
nn.Sequential(
    nn.Linear(hidden_dim, internal_dim),
    nn.ReLU(),
    nn.Linear(internal_dim, internal_dim // 2),
    nn.ReLU(),
    nn.Linear(internal_dim // 2, 1),
    nn.Tanh()  # Bound to [-1, 1]
)
```

**Key Property**: This network's output is NOT backpropagated through for task loss. It's only used to compute x₁₂ based on the hidden state.

### 4. Prediction Network

**Purpose**: Predict next hidden state to compute surprise.

**Architecture**:
```python
nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim)
)
```

**Usage**:
```python
predicted_next = predictor(h(t))
actual_next = encoder(s(t+1))
surprise = MSE(predicted_next, actual_next)
```

### 5. Attention Network

**Purpose**: Compute attention signal for x₁₂.

**Architecture**:
```python
nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.Tanh(),
    nn.Linear(hidden_dim // 2, 1),
    nn.Sigmoid()  # Bound to [0, 1]
)
```

### 6. Modulation Network

**Purpose**: Transform [x₁₂, m₁₂] into modulation signal.

**Architecture**:
```python
# Main modulation
nn.Sequential(
    nn.Linear(2, hidden_dim // 2),
    nn.Tanh(),
    nn.Linear(hidden_dim // 2, hidden_dim),
    nn.Tanh()
)

# Gating mechanism
nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
```

**Application**:
```python
modulation = modulation_network([x12, m12])
strength = gate([x12, m12])
h_mod = h + strength * modulation
```

### 7. Policy and Value Heads

**Purpose**: Generate actions and value estimates.

**Architecture**:
```python
# Policy
nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.Linear(hidden_dim // 2, output_dim)
)

# Value
nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.Linear(hidden_dim // 2, 1)
)
```

---

## Data Flow

### Forward Pass (Inference)

1. **Encode state**:
   ```python
   h = encoder(state)
   ```

2. **Optional LSTM**:
   ```python
   if use_lstm:
       h, (h_state, c_state) = lstm(h, (h_state, c_state))
   ```

3. **Compute attention**:
   ```python
   attention = attention_network(h)
   ```

4. **Get current internal state**:
   ```python
   x12 = internal_state.x12
   m12 = internal_state.m12
   ```

5. **Compute modulation**:
   ```python
   modulation = modulation_network([x12, m12])
   strength = gate([x12, m12])
   h_mod = h + strength * modulation
   ```

6. **Generate outputs**:
   ```python
   policy_logits = policy_head(h_mod)
   value = value_head(h_mod)
   ```

### Update Pass (After Action Taken)

1. **Compute surprise**:
   ```python
   predicted_next = predictor(h_current)
   actual_next = encoder(state_next)
   surprise = MSE(predicted_next, actual_next)
   ```

2. **Compute novelty**:
   ```python
   novelty = -log(count(state_next) / total_states)
   ```

3. **Update x₁₂**:
   ```python
   x12 = tanh(α·surprise + β·novelty + γ·attention - δ·x12)
   ```

4. **Update m₁₂** (if reward available):
   ```python
   importance = |reward|
   m12 = tanh(m12 + η·x12·importance - ζ·(m12 - baseline))
   ```

---

## Training Procedure

### Standard RL Training (Baseline)

```python
for episode in episodes:
    state = env.reset()

    for step in episode:
        # Forward pass
        policy, value = network(state)

        # Sample action
        action = sample(policy)

        # Environment step
        next_state, reward, done = env.step(action)

        # Compute loss
        policy_loss = compute_policy_loss(...)
        value_loss = compute_value_loss(...)
        total_loss = policy_loss + value_loss

        # Backprop
        total_loss.backward()
        optimizer.step()
```

### Internal Dimension Training (IDN)

```python
for episode in episodes:
    state = env.reset()
    network.reset_lstm()  # Reset LSTM, keep m12

    for step in episode:
        # Forward pass
        policy, value, internals = network(state, return_internals=True)
        h_current = internals['hidden']

        # Sample action
        action = sample(policy)

        # Environment step
        next_state, reward, done = env.step(action)

        # === UPDATE INTERNAL DIMENSIONS ===
        update_info = network.update_internal_state(
            current_hidden=h_current,
            next_state=next_state,
            reward=reward
        )

        # Optional: Add intrinsic reward
        intrinsic = network.compute_intrinsic_reward(method='balanced')
        total_reward = reward + 0.1 * intrinsic  # Weighted sum

        # Compute loss (using total_reward)
        policy_loss = compute_policy_loss(...)
        value_loss = compute_value_loss(...)

        # Optional: Add internal loss
        # (encourages x12 to be informative)
        x12_loss = -update_info['prediction_error']  # Reward accurate predictions

        total_loss = policy_loss + value_loss + 0.01 * x12_loss

        # Backprop (ONLY through external pathway)
        total_loss.backward()
        optimizer.step()

        # Log internal state
        logger.log({
            'x12': update_info['x12'],
            'm12': update_info['m12'],
            'surprise': update_info['prediction_error'],
            'novelty': update_info['novelty'],
        })
```

**Key Difference**: Internal dimensions are updated via their own dynamics, not directly via backprop from task loss.

---

## Design Decisions

### Why separate internal pathway?

**Alternative**: Train x₁₂ end-to-end with task loss.

**Problem**: x₁₂ would just learn to maximize reward, not to represent true "surprise."

**Solution**: x₁₂ evolves via prediction error, novelty, attention - intrinsic signals not dependent on task reward.

### Why tanh activation for internal dimensions?

**Reason**: Ensures boundedness x₁₂, m₁₂ ∈ [-1, 1].

**Benefit**: Interpretable ranges, prevents explosion, matches 12D CST formulation.

### Why modulation instead of direct input?

**Alternative**: Concatenate [h, x₁₂, m₁₂] and feed to policy.

**Problem**: Network might ignore internal dimensions.

**Solution**: Modulation forces internal state to *transform* the hidden representation, ensuring it influences behavior.

### Why keep m₁₂ between episodes?

**Reason**: m₁₂ represents long-term memory/wisdom that should persist.

**Analogy**: Biological organisms don't forget all experiences when starting a new task.

### Why LayerNorm instead of BatchNorm?

**Reason**: RL often uses small batches or online updates.

**Benefit**: LayerNorm normalizes per-sample, works with batch size = 1.

### Network Size Guidelines

| Environment Complexity | input_dim | hidden_dim | internal_dim |
|------------------------|-----------|------------|--------------|
| Simple (GridWorld) | 4-16 | 64-128 | 32 |
| Medium (Atari) | 3×84×84 | 256-512 | 64-128 |
| Complex (Robotics) | 100+ | 512-1024 | 128-256 |

---

## Comparison to Standard RL

| Aspect | Standard RL | Internal Dimension RL |
|--------|-------------|----------------------|
| **State** | External only | External + Internal (x₁₂, m₁₂) |
| **Training Signal** | Task reward only | Task reward + Internal dynamics |
| **Memory** | None or LSTM | LSTM + m₁₂ (accumulated wisdom) |
| **Exploration** | ε-greedy or count-based | x₁₂-driven (curiosity) |
| **Generalization** | Task-specific | Enhanced via m₁₂ transfer |
| **Interpretability** | Black box | x₁₂/m₁₂ provide insight |
| **Consciousness Metrics** | N/A | R_ω, R_ψ, autonomy, etc. |

### When to Use Internal Dimension RL?

**Use IDN when**:
- Long-term memory is important
- Transfer learning across tasks
- Interpretability matters
- Exploring consciousness in AI
- Sparse rewards (intrinsic motivation helps)

**Use Standard RL when**:
- Simple, fully observable tasks
- Ample training data
- Computational efficiency critical
- No need for interpretability

---

## Extension Points

### 1. Multi-Agent Internal Dimensions

Each agent has its own x₁₂, m₁₂, but can observe others':

```python
modulation_input = [
    own_x12, own_m12,
    teammate_x12, teammate_m12,
    opponent_x12, opponent_m12
]
```

Enables:
- Empathy (responding to others' x₁₂)
- Trust (integrating others' m₁₂)
- Theory of mind

### 2. Hierarchical Internal Dimensions

Different time scales:
- x₁₂_fast, m₁₂_fast: Within episode
- x₁₂_slow, m₁₂_slow: Across episodes

```python
m12_fast += η_fast * x12_fast * importance
m12_slow += η_slow * m12_fast * meta_importance
```

### 3. Multiple Internal Dimensions

Beyond x₁₂ and m₁₂:
- x₁₃: Emotional valence
- m₁₃: Personality traits
- x₁₄: Social awareness

### 4. Attention over Internal History

Instead of just current x₁₂, m₁₂:

```python
internal_context = attention(query=h, keys=m12_history)
```

Enables:
- Episodic memory
- Context-dependent behavior
- Reminiscence

---

## Implementation Checklist

When implementing IDN:

- [ ] Encoder with LayerNorm
- [ ] Optional LSTM for temporal processing
- [ ] x₁₂ network (NOT backpropped from task loss)
- [ ] Prediction network for surprise
- [ ] Attention network
- [ ] Internal state manager (InternalDimensionState)
- [ ] Modulation network with gating
- [ ] Policy and value heads
- [ ] Update procedure for x₁₂ (surprise + novelty + attention)
- [ ] Integration procedure for m₁₂
- [ ] Intrinsic reward computation
- [ ] Consciousness metrics tracking
- [ ] Logging of internal states
- [ ] Checkpoint saving/loading with internal state

---

**Last Updated**: 2024-11-16
**Version**: 1.0
**Authors**: NavisWORLD Research Team
