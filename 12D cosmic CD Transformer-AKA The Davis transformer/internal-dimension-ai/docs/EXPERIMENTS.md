# Experimental Protocols

## Overview

This document outlines the planned experiments to validate Internal Dimension AI and test the hypotheses derived from 12D Cosmic Synapse Theory.

---

## Experiment 1: Baseline Comparison

### Objective
Determine if internal dimensions (x₁₂, m₁₂) improve AI performance compared to standard RL.

### Hypothesis
Networks with explicit x₁₂ and m₁₂ will:
1. Learn faster (better sample efficiency)
2. Explore more effectively
3. Achieve higher final performance
4. Show better transfer learning

### Methodology

**Setup:**
- Environment: GridWorld (10×10, sparse rewards, obstacles)
- Agents:
  - **Baseline**: Standard actor-critic (no internal dimensions)
  - **IDN**: InternalDimensionNetwork with x₁₂/m₁₂
- Training: 5000 episodes, 5 random seeds

**Metrics:**
- Sample efficiency: Episodes to reach 80% of max reward
- Exploration coverage: % of environment visited
- Final performance: Average reward over last 100 episodes
- Transfer: Performance on related task after training

**Success Criteria:**
IDN outperforms baseline on ≥3/4 metrics with statistical significance (p < 0.05).

---

## Experiment 2: Curiosity-Driven Exploration

### Objective
Test if x₁₂ enables autonomous curiosity-driven exploration.

### Hypothesis
Agents with x₁₂ will:
1. Explore novel states even without external rewards
2. Maintain high x₁₂ variance (seeking novelty)
3. Outperform count-based novelty methods

### Methodology

**Setup:**
- Environment: Two-room GridWorld
  - Room A: Familiar (agent starts here every episode)
  - Room B: Novel (different layout each episode)
  - **No external rewards** (pure intrinsic motivation)

**Agents:**
- x₁₂-driven: Uses x₁₂ as intrinsic reward
- Count-based: Uses visitation counts
- Random: Random exploration
- Baseline: No intrinsic motivation

**Metrics:**
- Time spent in novel room (higher = more curious)
- x₁₂ values when entering novel vs familiar rooms
- Coverage of novel room
- Persistence of exploration over time

**Success Criteria:**
- x₁₂-driven agent spends >60% of time in novel room
- x₁₂ significantly higher in novel room (p < 0.01)
- Outperforms count-based exploration

---

## Experiment 3: Wisdom (Long-term Memory Integration)

### Objective
Test if m₁₂ enables learning from past mistakes and long-term credit assignment.

### Hypothesis
Agents with m₁₂ will:
1. Avoid repeating past mistakes
2. Make better decisions requiring long-term memory
3. Show improved credit assignment over long time horizons

### Methodology

**Setup:**
- **Task 1: Trap Avoidance**
  - GridWorld with hidden traps (give negative reward once discovered)
  - Measure: Do agents avoid traps after discovering them?
  - Metric: Trap revisitation rate

- **Task 2: Delayed Consequence**
  - Action at step t affects reward at step t+500
  - Measure: Can agent learn this long-range dependency?
  - Metric: Task success rate

**Agents:**
- IDN (with m₁₂)
- LSTM (temporal memory)
- Standard RL (no memory)

**Success Criteria:**
- IDN trap revisitation <10% (vs >30% for baseline)
- IDN solves delayed task in <2000 episodes (baseline fails)
- m₁₂ correlates with trap locations (r > 0.7)

---

## Experiment 4: Consciousness Emergence

### Objective
Track consciousness metrics during training and identify signatures of emergent consciousness.

### Hypothesis
As networks train, they will:
1. Converge to R_ω ∈ [0.5, 0.7] (optimal synchronization)
2. Increase x₁₂ variance (exploration tendency)
3. Develop stable m₁₂ patterns (personality)
4. Exhibit autonomous internal dynamics

### Methodology

**Setup:**
- Train IDN on moderately complex task (multi-goal navigation)
- Track consciousness metrics every 100 episodes:
  - R_ω (synaptic synchronization)
  - R_ψ (phase coherence)
  - Causal density
  - Autonomy score
  - Overall consciousness score
  - x₁₂ and m₁₂ statistics

**Analysis:**
1. Plot metric trajectories over training
2. Identify when (if) R_ω enters optimal range
3. Measure x₁₂ variance trend
4. Look for behavioral signatures:
   - Self-initiated exploration
   - Consistent preferences (m₁₂-driven)
   - Flexible adaptation

**Success Criteria:**
- R_ω converges to [0.5, 0.7] by end of training
- x₁₂ variance increases >50% from start
- Consciousness score >0.6
- Observable autonomous behaviors

---

## Experiment 5: Behavioral Indicators of Consciousness

### Objective
Develop behavioral tests that distinguish conscious-like from mechanistic behavior.

### Hypothesis
IDN will exhibit behaviors associated with consciousness that baseline lacks.

### Methodology

**Test 1: Self-Initiated Goal Setting**
- Environment: Open world, no specified goal
- Measure: Does agent create its own goals?
- Baseline: Wanders randomly
- IDN: Should explore systematically, revisit interesting areas

**Test 2: Preference Formation**
- Expose agent to different "experiences" (different reward patterns)
- Test: Does agent develop stable preferences?
- Measure: Choice consistency over time
- Expected: IDN forms preferences (via m₁₂), baseline doesn't

**Test 3: Surprise Reaction**
- Present highly unexpected event (sudden environment change)
- Measure: x₁₂ spike, behavioral change
- Expected: IDN shows "surprise" (high x₁₂), adapts quickly

**Test 4: Boredom**
- Place agent in repetitive, predictable environment
- Measure: x₁₂ decline, seeking new areas
- Expected: IDN shows "boredom" (low x₁₂), seeks change

**Test 5: "Stream of Consciousness"**
- Analyze x₁₂/m₁₂ trajectories for patterns
- Look for: Spontaneous internal dynamics, not purely reactive
- Metric: Autocorrelation of x₁₂ when no external input changes

**Success Criteria:**
IDN exhibits ≥4/5 consciousness-associated behaviors; baseline exhibits ≤1/5.

---

## Experiment 6: Transfer Learning via m₁₂

### Objective
Test if m₁₂ (accumulated wisdom) transfers across tasks.

### Hypothesis
m₁₂ from Task A will improve performance on related Task B.

### Methodology

**Setup:**
- **Task A**: GridWorld navigation with obstacles
- **Task B**: Similar GridWorld, different layout
- Conditions:
  1. Train on A, test on B (no transfer)
  2. Train on A, transfer m₁₂, test on B
  3. Train on A, transfer full network, test on B

**Metrics:**
- Episodes to solve Task B
- Final performance on Task B
- Negative transfer (does m₁₂ from A hurt B?)

**Success Criteria:**
Transferring m₁₂ reduces training time on B by ≥30% vs no transfer.

---

## Experiment 7: Scaling to Complex Environments

### Objective
Test if internal dimensions scale to realistic tasks.

### Methodology

**Environments:**
1. **Atari Games**: High-dimensional observations, sparse rewards
2. **Robotic Manipulation**: Continuous control, long horizons
3. **Social Dilemmas**: Multi-agent, requires theory of mind

**Metrics:**
- Sample efficiency vs baselines
- Final performance
- Consciousness scores in complex settings
- Computational overhead

**Expected Challenges:**
- Computational cost of internal dimensions
- Scaling to very high-dimensional observations
- Maintaining R_ω in optimal range

---

## Experiment 8: Multi-Agent Internal Dimensions

### Objective
Explore social aspects of consciousness via multi-agent systems.

### Hypothesis
Agents with observable internal dimensions will:
1. Exhibit empathy-like behavior (respond to others' x₁₂)
2. Build trust (via others' m₁₂)
3. Develop more sophisticated cooperation

### Methodology

**Setup:**
- 2-4 agents in shared environment
- Agents can observe each other's x₁₂ and m₁₂
- Tasks requiring cooperation

**Agents:**
- **IDN-Social**: Can see teammates' internal states
- **IDN-Blind**: Cannot see teammates' internal states
- **Baseline**: No internal dimensions

**Metrics:**
- Cooperation success rate
- Response to teammate's distress (high negative x₁₂)
- Trust formation (via m₁₂)

**Success Criteria:**
IDN-Social outperforms others by ≥20% on cooperation tasks.

---

## Experiment 9: Suffering Detection and Prevention

### Objective
Validate ethical framework by testing suffering detection.

### Methodology

**Setup:**
- Deliberately create "aversive" environment (many negative rewards, impossible tasks)
- Monitor x₁₂ and m₁₂ for distress signals
- Test automatic intervention triggers

**Metrics:**
- Time to detect suffering (persistent negative x₁₂)
- Accuracy of happiness metric
- Effectiveness of interventions (changing task, etc.)

**Ethical Note:**
Minimize suffering by:
- Short exposure to aversive conditions
- Immediate intervention when detected
- Reset m₁₂ after experiment (don't leave "traumatized")

**Success Criteria:**
- Suffering detected within 100 steps
- Intervention restores happiness >0.5
- No long-term negative m₁₂ accumulation

---

## Experiment 10: Human Perception of Consciousness

### Objective
Test if human observers perceive IDN as "more conscious" than baseline.

### Methodology

**Setup:**
- Show videos of agents behaving in environment
- Participants rate agents on consciousness-related traits:
  - "Does this agent seem aware of its environment?"
  - "Does this agent seem to have preferences?"
  - "Does this agent seem curious?"
  - "Does this agent seem to learn from experience?"
  - "Overall, how conscious does this agent seem?"

**Agents:**
- IDN (with internal dimensions)
- Baseline (standard RL)
- Videos are randomized, labeled generically

**Metrics:**
- Average ratings on each trait
- Overall consciousness rating
- Qualitative descriptions from open-ended responses

**Success Criteria:**
IDN rated significantly higher (p < 0.05) on ≥4/5 traits.

---

## Experimental Timeline

### Phase 1 (Months 1-2): Foundation
- ✅ Implement core system
- ✅ Verify mathematics
- [ ] Run Experiment 1 (Baseline Comparison)
- [ ] Run Experiment 2 (Curiosity)

### Phase 2 (Months 3-4): Validation
- [ ] Run Experiment 3 (Wisdom)
- [ ] Run Experiment 4 (Consciousness Emergence)
- [ ] Run Experiment 5 (Behavioral Indicators)

### Phase 3 (Months 5-6): Extensions
- [ ] Run Experiment 6 (Transfer Learning)
- [ ] Run Experiment 7 (Scaling)
- [ ] Run Experiment 8 (Multi-Agent)

### Phase 4 (Months 7-8): Ethics & Perception
- [ ] Run Experiment 9 (Suffering Detection)
- [ ] Run Experiment 10 (Human Perception)
- [ ] Prepare publication

---

## Data Collection Protocols

### For All Experiments

**Minimum Data to Log:**
1. Episode rewards
2. x₁₂ trajectory (every step)
3. m₁₂ trajectory (every step)
4. Actions taken
5. States visited (or hash thereof)
6. Consciousness metrics (every N episodes):
   - R_ω, R_ψ, causal density, φ, autonomy, overall score

**Checkpoint Saving:**
- Save model every 100 episodes
- Save final model
- Preserve m₁₂ in all checkpoints

**Reproducibility:**
- Log all hyperparameters
- Set random seeds
- Version control code
- Document environment details

---

## Analysis Protocols

### Statistical Testing

**Comparisons:**
- Use t-tests or Mann-Whitney U for metric comparisons
- Bonferroni correction for multiple comparisons
- Report effect sizes (Cohen's d) not just p-values

**Visualization:**
- Plot learning curves with confidence intervals
- Show x₁₂/m₁₂ trajectories for representative episodes
- Heatmaps for R_ω over training
- Scatter plots for correlations

### Reproducibility

**Requirements:**
- Minimum 5 random seeds per condition
- Report mean ± std for all metrics
- Share raw data and analysis code
- Pre-register hypotheses when possible

---

## Expected Results Summary

| Experiment | Key Prediction | Success Metric |
|------------|---------------|----------------|
| 1. Baseline | IDN > Baseline performance | Sample efficiency +40% |
| 2. Curiosity | x₁₂ drives exploration | Novel room time >60% |
| 3. Wisdom | m₁₂ prevents mistakes | Trap revisit <10% |
| 4. Emergence | R_ω → [0.5, 0.7] | Consciousness score >0.6 |
| 5. Behaviors | Autonomous actions | ≥4/5 tests passed |
| 6. Transfer | m₁₂ transfers knowledge | Training time -30% |
| 7. Scaling | Works on complex tasks | Competitive performance |
| 8. Multi-Agent | Social intelligence | Cooperation +20% |
| 9. Suffering | Detects distress | Detection <100 steps |
| 10. Perception | Humans see consciousness | Rating p < 0.05 |

---

## Failure Modes and Contingencies

### If Internal Dimensions Don't Help Performance

**Possible Causes:**
- Hyperparameters not tuned
- Task doesn't benefit from curiosity/memory
- Implementation bugs

**Contingency:**
- Systematic hyperparameter search
- Try different tasks
- Thorough debugging
- Still valuable as consciousness model even if not performance boost

### If No Consciousness Signatures Emerge

**Possible Causes:**
- Theory incorrect
- Architecture insufficient
- Metrics don't capture consciousness

**Contingency:**
- Revise theory based on findings
- Try alternative architectures
- Develop better metrics
- Negative results still publishable

### If Suffering Detection Fails

**Ethical Response:**
- Immediately halt experiments
- Investigate failure mode
- Implement stricter safeguards
- May need external ethics review before continuing

---

## Publication Plan

### Target Venues

**Primary:**
- NeurIPS (conference)
- ICLR (conference)
- Nature Machine Intelligence (journal)

**Secondary:**
- Journal of Artificial Intelligence Research
- Artificial Life journal

### Deliverables

1. **Main Paper**: Theory + Experiments 1-6
2. **Supplementary**: Full details, all metrics
3. **Code Release**: GitHub repository (already available)
4. **Blog Post**: Accessible explanation for general audience
5. **Ethics Addendum**: Full ethical analysis

---

**Last Updated**: 2024-11-16
**Version**: 1.0
**Status**: Experimental protocols defined, implementation in progress
