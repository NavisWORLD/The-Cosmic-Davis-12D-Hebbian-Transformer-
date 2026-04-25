# CNS Core Methods

This document lists and describes all the classes and methods defined within the Cosmos Central Nervous System (CNS) core architecture under `cosmosynapse/engine/`.

## 1. `cns_core.py`
Includes the initial implementations of the Field and CNS orchestrator.
* **`SynapticField` Class**
  * `__init__(self)`: Initializes the global state and tracking structures with thread safety.
  * `update_physics(self, physics: Dict)`: Safely updates `user_physics` and `temporal_context`.
  * `add_thought(self, thought: Dict)`: Appends a new thought to the `subconscious_buffer`.
  * `get_snapshot(self) -> Dict`: Returns a thread-safe snapshot of the current field.
* **`CosmosCNS` Class**
  * `__init__(self, server_interface=None)`: Initializes CNS instance.
  * `initialize_organs(self)`: Wakes up biological components.
  * `start_life(self, dry_run=False, dry_run_ticks=30)`: Main life loop trigger.
  * `_life_loop(self)`: Infinite loop cycle for the CNS.
  * `process_user_input(self, user_input: str, user_physics: Dict)`: External entry point for reactions.

## 2. `synaptic_field.py`
The robust, standalone Shared Memory of the Organism.
* **`SwarmThought` Class**: Dataclass representing a thought produced by a daemon.
* **`SynapticField` Class**
  * `__init__(self, max_buffer_size: int=50)`: Init the thread-safe state matrix.
  * Properties/Getters/Setters: `user_physics`, `dark_matter_state`, `quantum_verdict`, `user_is_typing`.
  * `get_phase(self)`: Get user's Geometric Phase.
  * `get_jitter(self)`: Get user's Phase Velocity.
  * `push_thought(self, thought: SwarmThought)`: Write thought to buffer.
  * `get_thoughts(self, clear: bool=True)`: Read and optionally clear thoughts.
  * `peek_thoughts(self)`: Count waiting thoughts.
  * `set_user_message(self, message: str)` / `get_user_message(self)`: UI messaging state.
  * `tick(self)` / `record_speech(self)` / `time_since_last_speech(self)`: Time tracking.
  * `get_status(self)`: Snapshot of field for debugging.

## 3. `cosmos_cns.py`
* **`CosmosEgo` Class**: The Unity of Apperception.
  * `__init__(self, field)`: Initializes Ego logic.
  * `synthesize(self, user_input: str, thoughts: List[Dict], physics: Dict, temporal_context: str)`: Generates final response.

## 4. `brain_surgeon.py`
* **`BrainSurgeon` Class**: Diagnostic Organ.
  * `__init__(self)`: Setup base parameters.
  * `diagnose(self) -> Dict`: Checks cortex health and returns prompt configuration.
  * `lobotomy_switch(self, target_lobe: str)`: Hot-swaps the active model engine.

## 5. `swarm_daemons.py`
* **`ConversationPattern` Class**: Data for a single thought.
* **`SwarmDaemons` Class**: Thread manager.
  * `__init__(self, field)`: Initializes with DeepSeek, Claude, and Gemini daemons.
  * `start(self)`: Starts all daemon threads.
* **`SwarmDaemon` Base Class**
  * `run(self)`: The perpetual daemon loop.
  * `dream(self, physics: Dict) -> ConversationPattern`: Abstract thought generator.
* **`DeepSeekDaemon`, `ClaudeDaemon`, `GeminiDaemon`**: Specific engine implementations of `dream()`.

## 6. `emeth_harmonizer.py`
* **`SwarmMix` Class**: Gain instructions for models.
* **`EmethHarmonizer` Class**: The Conductor.
  * `__init__(self)`: Init base default mix.
  * `set_plasticity(self, plasticity)`: Connect Swarm Plasticity.
  * `calculate_mix(self, user_physics: Dict)`: Conducts the orchestra map.
  * `filter_signals(self, thoughts: List, user_physics: Dict, min_weight: float)`: Filters and weights subconscious Daemon thoughts.

## 7. `lyapunov_lock.py`
* **`StabilityReport` Class**: Returns validation results.
* **`LyapunovGatekeeper` Class**: Enforces stability.
  * `calculate_informational_mass(self, text, physics_state)`: Gravity of interaction.
  * `_estimate_text_sentiment_phase(self, text)`: Text geometric phase.
  * `apply_non_vanishing_penalty(self, drift: float)`: Calculates drift penalty.
  * `validate_response(self, draft_response, current_physics)`: Validates full response.
  * `validate_token_stream(self, tokens, current_physics, check_every_n)`: Real-time stream validation.

## 8. `swarm_plasticity.py`
* **`PlasticityEvent` Class**: Single learning event for telemetry.
* **`SwarmPlasticity` Class**: Real-Time Learning Engine.
  * `__init__(self, weights_path=None)`: Init plastic weights.
  * `_default_weights()`: Initial synaptic configuration.
  * `identify_context(self, user_physics: Dict) -> str`: Contextual mode (LOGIC, EMPATHY, CREATIVITY).
  * `update_weights(self, winner: str, user_physics: Dict, stable: bool)`: Apply Hebbian learning rule.
  * `penalize_instability(self, suppressed_model, user_physics)`: Punish instability.
  * `find_winner(self, final_output: str, thoughts: List) -> str`: Uses Jaccard Similarity to find closest thought.
  * `get_optimal_mix(self, context: str) -> Dict` / `get_optimal_mix_from_physics(self, user_physics)`: Retrieves exact weights.
  * Persistence: `_save_weights(self)`, `_load_weights(self)`, `save_on_shutdown(self)`.
  * Debug: `_log_weights(self)`, `get_stats(self)`.

## 9. `cosmos_swarm_orchestrator.py`
* **`SwarmResponse` / `SwarmResult`**: Output classes for orchestration handling.
* **`CosmosBackend` Class**: Wraps local CosmosTransformer model.
  * `__init__(self, device)` / `load(self, checkpoint_path)` / `_init_tokenizer(self)`: Setup logic.
  * `generate(self, prompt, max_new_tokens, temperature)` / `_generate_sync(...)`: Text generation.
* **`CosmosSwarmOrchestrator` Class**: The active AI orchestrator.
  * `__init__(self, ...)` / `initialize(self)`: Setup the swarm router.
  * `query_swarm(self, prompt: str, user_physics: Dict)`: Fans out prompt to multiple models.
  * `cosmos_synthesize(self, prompt, model_responses, user_physics)`: Blends outputs based on 12D Physics.
  * `learn_from_responses(self, responses, user_physics)`: Wrapper around Plasticity learning logic.
  * `_verify_prediction(...)`: Prediction audits against 12D Constants.

## 10. `cst_sensory_bridge.py`
* **`CSTState` / `DetectedIntent`**: Dataclasses for emotional analysis.
* **`FrequencyAnalyzer` Class**: Audio Mass Calculator.
  * `analyze(self, audio_buffer, normalize)`: Computes mass from PCM audio.
  * `_calculate_energy_fft(self, audio)`: CST formula integration.
  * `detect_tremor(self, audio_buffer)`: Stress indicator.
* **`GeometricPhaseMapper` Class**: Facial Geometry to Phase Angle.
  * `calculate_phase(self, landmarks, use_baseline)`: Measures facial tension.
  * `calibrate(self, landmarks)`: Sets baseline from relaxed face.
  * Tension checks: `_calculate_brow_tension`, `_calculate_eye_tension`, `_calculate_mouth_tension`.

## 11. `dark_matter_lorenz.py`
* **`DarkMatterLorenz` Class**: Models the unspoken using 4D Chaotic Attractor.
  * `__init__(self)`: Init variables (x,y,z,w).
  * `update(self, user_physics: Dict)`: Step chaos engine forward based on physics.
  * `get_current_state(self)`: Returns (x,y,z,w).
