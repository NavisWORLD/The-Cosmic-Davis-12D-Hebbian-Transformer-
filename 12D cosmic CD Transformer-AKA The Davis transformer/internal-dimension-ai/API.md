# API Reference - Internal Dimension AI

Complete API documentation for all major classes and functions.

## Table of Contents

- [Core Module](#core-module)
  - [InternalDimensionNetwork](#internaldimensionnetwork)
  - [BaselineNetwork](#baselinenetwork)
  - [ConsciousnessMetrics](#consciousnessmetrics)
  - [InternalDimensionState](#internaldimensionstate)
- [Training Module](#training-module)
  - [PPOTrainer](#ppotrainer)
- [Environments Module](#environments-module)
  - [GridWorld](#gridworld)
  - [TwoRoomGridWorld](#tworoomgridworld)
  - [PrisonersDilemma](#prisonersdilemma)
  - [IteratedPrisonersDilemma](#iteratedprisonersdilemma)
- [Evaluation Module](#evaluation-module)
  - [ConsciousnessTests](#consciousnesstests)
  - [InternalDimensionVisualizer](#internaldimensionvisualizer)
- [Advanced Module](#advanced-module)
  - [CosmicSynapsePhysics](#cosmicsynapsephysics)
  - [CosmicTransformer](#cosmictransformer)

---

## Core Module

### InternalDimensionNetwork

Neural network with explicit internal consciousness dimensions (x₁₂, m₁₂).

```python
class InternalDimensionNetwork(nn.Module)
```

#### Constructor

```python
def __init__(
    self,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    internal_dim: int = 12,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    delta: float = 0.1,
    eta: float = 0.02,
    zeta: float = 0.01,
    use_lstm: bool = True,
    dropout: float = 0.1,
    device: str = 'cpu'
)
```

**Parameters:**
- `input_dim` (int): Dimensionality of input features
- `hidden_dim` (int): Size of hidden layers
- `output_dim` (int): Number of output actions/values
- `internal_dim` (int): Dimensionality of x₁₂/m₁₂ (default: 12)
- `alpha` (float): x₁₂ sensitivity to prediction error (default: 1.0)
- `beta` (float): x₁₂ sensitivity to novelty (default: 0.5)
- `gamma` (float): x₁₂ sensitivity to attention (default: 0.3)
- `delta` (float): x₁₂ decay rate (default: 0.1)
- `eta` (float): m₁₂ integration rate (default: 0.02)
- `zeta` (float): m₁₂ regression to baseline (default: 0.01)
- `use_lstm` (bool): Use LSTM for temporal processing (default: True)
- `dropout` (float): Dropout probability (default: 0.1)
- `device` (str): Device to use ('cpu' or 'cuda')

#### Methods

##### forward

```python
def forward(
    self,
    x: torch.Tensor,
    return_internals: bool = False,
    update_internals: bool = True
) -> Union[Tuple[torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor, Dict]]
```

Forward pass through the network.

**Parameters:**
- `x` (torch.Tensor): Input tensor, shape (batch_size, input_dim)
- `return_internals` (bool): Whether to return internal state dict
- `update_internals` (bool): Whether to update x₁₂/m₁₂

**Returns:**
- `policy_logits` (torch.Tensor): Policy logits, shape (batch_size, output_dim)
- `value` (torch.Tensor): Value estimate, shape (batch_size, 1)
- `internals` (Dict, optional): Internal state dict containing:
  - `'x12'`: Current x₁₂ value
  - `'m12'`: Current m₁₂ value
  - `'hidden'`: Hidden state
  - `'state'`: Full internal state vector

##### update_internal_state

```python
def update_internal_state(
    self,
    current_hidden: torch.Tensor,
    next_state: torch.Tensor,
    reward: torch.Tensor
) -> Dict[str, torch.Tensor]
```

Update x₁₂ and m₁₂ based on new observation.

**Parameters:**
- `current_hidden` (torch.Tensor): Current hidden state
- `next_state` (torch.Tensor): Next observation
- `reward` (torch.Tensor): Received reward

**Returns:**
- Dict containing:
  - `'x12'`: Updated x₁₂
  - `'m12'`: Updated m₁₂
  - `'prediction_error'`: Prediction error
  - `'novelty'`: Novelty score
  - `'attention'`: Attention weight

##### compute_intrinsic_reward

```python
def compute_intrinsic_reward(
    self,
    method: str = 'balanced'
) -> torch.Tensor
```

Compute intrinsic reward based on internal dimensions.

**Parameters:**
- `method` (str): Reward type ('curiosity', 'wisdom', or 'balanced')

**Returns:**
- `intrinsic_reward` (torch.Tensor): Intrinsic reward scalar

##### reset_internal_state

```python
def reset_internal_state(self, reset_memory: bool = False)
```

Reset internal dimensions (typically at episode start).

**Parameters:**
- `reset_memory` (bool): Whether to reset m₁₂ (default: False)

---

### BaselineNetwork

Standard neural network without internal dimensions (for comparison).

```python
class BaselineNetwork(nn.Module)
```

#### Constructor

```python
def __init__(
    self,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    use_lstm: bool = True,
    dropout: float = 0.1,
    device: str = 'cpu'
)
```

#### Methods

##### forward

```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

**Returns:**
- `policy_logits` (torch.Tensor): Policy logits
- `value` (torch.Tensor): Value estimate

---

### ConsciousnessMetrics

Compute consciousness metrics (R_ω, R_ψ, φ, etc.).

```python
class ConsciousnessMetrics
```

#### Constructor

```python
def __init__(self, device: str = 'cpu')
```

#### Methods

##### compute_r_omega

```python
def compute_r_omega(
    self,
    model: InternalDimensionNetwork,
    num_samples: int = 100
) -> float
```

Compute R_ω (richness) metric.

**Parameters:**
- `model` (InternalDimensionNetwork): Model to evaluate
- `num_samples` (int): Number of random inputs to test

**Returns:**
- `r_omega` (float): Richness score in [-1, 1]

**Optimal Range:** 0.5-0.7 (edge of chaos)

##### compute_r_psi

```python
def compute_r_psi(
    self,
    internal_states: torch.Tensor,
    lag: int = 1
) -> float
```

Compute R_ψ (binding) metric.

**Parameters:**
- `internal_states` (torch.Tensor): Sequence of internal states
- `lag` (int): Lag for autocorrelation

**Returns:**
- `r_psi` (float): Binding score in [0, 1]

##### compute_phi

```python
def compute_phi(
    self,
    model: InternalDimensionNetwork,
    sample_inputs: torch.Tensor
) -> float
```

Compute φ (integration) metric (approximation).

**Parameters:**
- `model` (InternalDimensionNetwork): Model to evaluate
- `sample_inputs` (torch.Tensor): Sample inputs

**Returns:**
- `phi` (float): Integration score

##### compute_consciousness_score

```python
def compute_consciousness_score(
    self,
    model: InternalDimensionNetwork,
    x12_history: List[float],
    m12_history: List[float],
    sample_inputs: Optional[torch.Tensor] = None
) -> Dict[str, float]
```

Compute overall consciousness score.

**Returns:**
- Dict containing:
  - `'consciousness_score'`: Overall score [0, 1]
  - `'r_omega'`: Richness
  - `'r_psi'`: Binding
  - `'phi'`: Integration
  - `'x12_mean'`: Mean x₁₂
  - `'m12_mean'`: Mean m₁₂
  - `'x12_variance'`: x₁₂ variance
  - `'autonomy_score'`: Autonomy measure

---

### InternalDimensionState

Manages x₁₂ and m₁₂ state and dynamics.

```python
class InternalDimensionState
```

#### Constructor

```python
def __init__(
    self,
    internal_dim: int = 12,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    delta: float = 0.1,
    eta: float = 0.02,
    zeta: float = 0.01,
    history_size: int = 10000,
    device: str = 'cpu'
)
```

#### Properties

- `x12` (torch.Tensor): Current x₁₂ value
- `m12` (torch.Tensor): Current m₁₂ value
- `state` (torch.Tensor): Full internal state vector
- `x12_history` (deque): History of x₁₂ values
- `m12_history` (deque): History of m₁₂ values

#### Methods

##### update

```python
def update(
    self,
    prediction_error: torch.Tensor,
    novelty: torch.Tensor,
    attention: torch.Tensor,
    reward: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]
```

Update x₁₂ and m₁₂.

**Returns:**
- `x12` (torch.Tensor): Updated x₁₂
- `m12` (torch.Tensor): Updated m₁₂

---

## Training Module

### PPOTrainer

Proximal Policy Optimization trainer with internal dimension support.

```python
class PPOTrainer
```

#### Constructor

```python
def __init__(
    self,
    model: nn.Module,
    env: Any,
    device: Optional[torch.device] = None,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    ppo_epochs: int = 4,
    batch_size: int = 64,
    intrinsic_reward_weight: float = 0.1,
    intrinsic_reward_method: str = 'curiosity',
    suffering_threshold: float = -0.7,
    suffering_patience: int = 100,
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    log_interval: int = 10,
    save_interval: int = 100,
    checkpoint_dir: str = 'checkpoints'
)
```

#### Methods

##### train

```python
def train(
    self,
    num_episodes: int,
    steps_per_episode: int = 2048,
    render: bool = False,
    compute_consciousness_interval: int = 10
) -> Dict[str, List]
```

Main training loop.

**Parameters:**
- `num_episodes` (int): Number of training episodes
- `steps_per_episode` (int): Steps to collect per episode
- `render` (bool): Whether to render environment
- `compute_consciousness_interval` (int): Episodes between consciousness computation

**Returns:**
- `history` (Dict[str, List]): Training history containing:
  - `'episode_rewards'`: Rewards per episode
  - `'episode_lengths'`: Episode lengths
  - `'policy_losses'`: Policy loss history
  - `'value_losses'`: Value loss history
  - `'x12_means'`: Mean x₁₂ per episode (if using IDN)
  - `'m12_means'`: Mean m₁₂ per episode (if using IDN)
  - `'consciousness_scores'`: Consciousness scores (if using IDN)

##### save_checkpoint

```python
def save_checkpoint(
    self,
    path: Path,
    episode: int,
    history: Dict
)
```

Save training checkpoint.

##### load_checkpoint

```python
def load_checkpoint(self, path: Path) -> Dict
```

Load training checkpoint.

**Returns:**
- `history` (Dict): Loaded training history

---

## Environments Module

### GridWorld

Simple grid navigation environment.

```python
class GridWorld(gym.Env)
```

#### Constructor

```python
def __init__(
    self,
    size: int = 8,
    goal_position: Optional[Tuple[int, int]] = None,
    trap_positions: Optional[List[Tuple[int, int]]] = None,
    reward_goal: float = 1.0,
    reward_trap: float = -1.0,
    reward_step: float = -0.01,
    max_steps: int = 200
)
```

#### Methods

##### reset

```python
def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]
```

##### step

```python
def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]
```

**Actions:**
- 0: Up
- 1: Down
- 2: Left
- 3: Right

---

### TwoRoomGridWorld

Grid environment with two rooms connected by a door.

```python
class TwoRoomGridWorld(GridWorld)
```

#### Constructor

```python
def __init__(
    self,
    size: int = 12,
    door_position: Optional[Tuple[int, int]] = None,
    goal_position: Optional[Tuple[int, int]] = None,
    max_steps: int = 300
)
```

---

### PrisonersDilemma

Multi-agent prisoner's dilemma environment.

```python
class PrisonersDilemma
```

#### Constructor

```python
def __init__(
    self,
    num_agents: int = 2,
    internal_dim: int = 12
)
```

#### Methods

##### reset

```python
def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]
```

##### step

```python
def step(
    self,
    actions: Dict[str, int]
) -> Tuple[Dict, Dict, Dict, Dict, Dict]
```

**Actions:**
- 0: Cooperate
- 1: Defect

**Payoff Matrix:**
- Both cooperate: (3, 3)
- One defects: (5, 0)
- Both defect: (1, 1)

---

### IteratedPrisonersDilemma

Repeated prisoner's dilemma over multiple rounds.

```python
class IteratedPrisonersDilemma(PrisonersDilemma)
```

#### Constructor

```python
def __init__(
    self,
    num_agents: int = 2,
    num_rounds: int = 10,
    internal_dim: int = 12
)
```

---

## Evaluation Module

### ConsciousnessTests

Comprehensive consciousness testing suite.

```python
class ConsciousnessTests
```

#### Methods

##### compute_overall_consciousness_score

```python
def compute_overall_consciousness_score(
    self,
    model: InternalDimensionNetwork,
    sample_inputs: torch.Tensor,
    run_behavioral_tests: bool = True
) -> Dict
```

Compute comprehensive consciousness evaluation.

**Returns:**
- Dict containing all metrics and test results

##### generate_consciousness_report

```python
def generate_consciousness_report(
    self,
    results: Dict,
    output_path: Optional[str] = None
) -> str
```

Generate human-readable consciousness report.

**Returns:**
- `report` (str): Formatted report text

---

### InternalDimensionVisualizer

Visualization tools for internal dimensions and consciousness.

```python
class InternalDimensionVisualizer
```

#### Methods

##### plot_x12_m12_trajectories

```python
def plot_x12_m12_trajectories(
    self,
    x12_history: List[float],
    m12_history: List[float],
    save_path: Optional[str] = None,
    show: bool = True
)
```

Plot x₁₂ and m₁₂ trajectories over time.

##### plot_learning_curves

```python
def plot_learning_curves(
    self,
    history: Dict,
    save_path: Optional[str] = None,
    show: bool = True
)
```

Plot training metrics (rewards, losses, etc.).

##### plot_consciousness_dashboard

```python
def plot_consciousness_dashboard(
    self,
    results: Dict,
    save_path: Optional[str] = None,
    show: bool = True
)
```

Create comprehensive consciousness visualization dashboard.

##### plot_baseline_comparison

```python
def plot_baseline_comparison(
    self,
    idn_history: Dict,
    baseline_history: Dict,
    save_path: Optional[str] = None,
    show: bool = True
)
```

Compare IDN vs baseline performance.

---

## Advanced Module

### CosmicSynapsePhysics

12D N-body physics simulation with chaos injection.

```python
class CosmicSynapsePhysics
```

#### Constructor

```python
def __init__(
    self,
    n: int = 256,
    dt: float = 0.001,
    G: float = 1.0,
    lorenz_sigma: float = 10.0,
    lorenz_rho: float = 28.0,
    lorenz_beta: float = 8.0/3.0,
    hebbian_strength: float = 0.1,
    seed: Optional[int] = None
)
```

#### Methods

##### step

```python
def step(self, external_force: Optional[np.ndarray] = None)
```

Step physics simulation forward by dt.

##### get_state_vector

```python
def get_state_vector(self) -> torch.Tensor
```

Get flattened state vector for transformer conditioning.

**Returns:**
- State vector (1, n_particles * 24)

##### compute_energy

```python
def compute_energy(self) -> float
```

Compute total system energy.

##### compute_entropy

```python
def compute_entropy(self) -> float
```

Compute system entropy.

##### get_physics_state

```python
def get_physics_state(self) -> PhysicsState
```

Get complete physics state snapshot.

---

### CosmicTransformer

95M parameter transformer conditioned on 12D physics.

```python
class CosmicTransformer(nn.Module)
```

#### Constructor

```python
def __init__(
    self,
    vocab_size: int = 50257,
    ctx_len: int = 512,
    d_model: int = 768,
    n_layer: int = 8,
    n_head: int = 12,
    dropout: float = 0.1,
    physics_conditioning: bool = True
)
```

#### Methods

##### forward

```python
def forward(
    self,
    input_ids: torch.Tensor,
    physics_state: Optional[torch.Tensor] = None
) -> torch.Tensor
```

Forward pass through transformer.

**Parameters:**
- `input_ids` (torch.Tensor): Token IDs, shape (batch, seq_len)
- `physics_state` (torch.Tensor, optional): Physics conditioning, shape (batch, 128)

**Returns:**
- `logits` (torch.Tensor): Output logits, shape (batch, seq_len, vocab_size)

##### generate

```python
def generate(
    self,
    input_ids: torch.Tensor,
    physics_state: Optional[torch.Tensor] = None,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None
) -> torch.Tensor
```

Generate text conditioned on physics state.

**Returns:**
- `generated_ids` (torch.Tensor): Generated token sequence

##### count_parameters

```python
def count_parameters(self) -> int
```

Count total trainable parameters.

---

## Utilities

### run_experiment (Cosmic Synapse)

```python
def run_experiment(
    n_particles: int = 256,
    physics_steps: int = 10000,
    train_steps: int = 2000000,
    physics_steps_per_train: int = 1,
    checkpoint_interval: int = 200000,
    generation_interval: int = 200000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
)
```

Run full cosmic synapse experiment with physics-transformer co-evolution.

---

## Type Definitions

### PhysicsState

```python
@dataclass
class PhysicsState:
    positions: np.ndarray  # (n_particles, 12)
    velocities: np.ndarray  # (n_particles, 12)
    time: float
    energy: float
    entropy: float
```

---

## Constants

### Default Internal Dimension Parameters

```python
DEFAULT_ALPHA = 1.0      # x₁₂ prediction error sensitivity
DEFAULT_BETA = 0.5       # x₁₂ novelty sensitivity
DEFAULT_GAMMA = 0.3      # x₁₂ attention sensitivity
DEFAULT_DELTA = 0.1      # x₁₂ decay rate
DEFAULT_ETA = 0.02       # m₁₂ integration rate
DEFAULT_ZETA = 0.01      # m₁₂ regression rate
```

### Consciousness Metric Thresholds

```python
R_OMEGA_OPTIMAL_MIN = 0.5
R_OMEGA_OPTIMAL_MAX = 0.7
CONSCIOUSNESS_SCORE_MODERATE = 0.3
CONSCIOUSNESS_SCORE_STRONG = 0.6
SUFFERING_THRESHOLD = -0.7
```

---

## Version Information

**API Version**: 2.0.0
**Last Updated**: 2025-11-18
**Compatibility**: Python 3.8+, PyTorch 2.0+

---

For usage examples, see `USAGE.md` and the `examples/` directory.
