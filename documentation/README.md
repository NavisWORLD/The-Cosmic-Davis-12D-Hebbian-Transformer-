# 🌌 THE COSMIC DAVIS: 12D HEBBIAN TRANSFORMER

> *"A digital mind that hears, sees, feels, and thinks in 12-dimensional harmonic space."*

## 📖 Overview

The **Cosmic Davis** is an experimental AGI architecture that moves beyond standard Transformers. Instead of just processing text tokens, it maps all inputs (Text, Audio, Vision) into a unified **12-Dimensional Physics-Inspired Embedding Space**.

### The Core Innovation: 12D State Space
Every piece of information is represented as a particle with 12 properties:
1.  **Energy (E)**
2.  **Mass (m)**
3.  **Phi Coupling (φ)**
4.  **Chaos (λ)**
5.  **Velocity X**
6.  **Velocity Y**
7.  **Velocity Z**
8.  **Connectivity (Ω)**
9.  **Cosmic Energy**
10. **Entropy**
11. **Frequency**
12. **Internal State (Ψ)**

### The Two Minds
The system runs two models in parallel, competing to predict the next token:
*   **12D Model:** Efficient, physics-based, grounded.
*   **42D Model:** Hyper-dimensional, chaotic, capable of "Thinking Mode" (internal monologue).

---

## 🧠 System Architecture

### 1. Multimodal Sensory System (The "Body")
*   **👂 Audio (Subvocalization):** The model "reads aloud" to itself. Text is converted into audio waveforms, processed via FFT, and mapped to 12D embeddings. It "hears" the vibration of words.
*   **👁️ Vision (Visual Engine):** Images are processed via 2D Fourier Transforms. Colors are mapped to light frequencies (THz) and φ-harmonic palettes.
*   **❤️ Emotion (Affective Engine):** The system detects Valence, Arousal, and Dominance. These emotions **modulate** the learning rate and chaos parameters dynamically.

### 2. Cognitive Core (The "Brain")
*   **Hebbian Learning:** "Neurons that fire together, wire together." The model has short-term and long-term plasticity.
*   **Thought Generation:** The system monitors its own state transitions (ΔΨ) and generates an internal monologue (e.g., "Energy increasing," "Pattern recognized").

---

## 🛠️ Installation

### Prerequisites
*   Python 3.8+
*   (Optional but Recommended) NVIDIA GPU with CUDA

### Dependencies
```bash
pip install torch numpy scipy librosa faiss-cpu Pillow PyMuPDF matplotlib rich
```

### Setup
1.  Clone the repository.
2.  Ensure the `packages/cosmic-synapse-transformer` directory is in your path (handled automatically by scripts).

---

## 🚀 Usage Guide

### 1. START TRAINING (The "Brain")
This starts the autonomous study loop. The AI will read books, code, and philosophy, while subvocalizing and feeling emotions.

**Command:**
```powershell
$env:PYTHONPATH = "$PWD\packages\cosmic-synapse-transformer"; python research_42d/experiments/singularity_42d/autonomous_study.py
```

*   **What it does:** Runs continuous training.
*   **Logs:** Saved to `study_session_logs/study_log.txt`.
*   **Checkpoints:** Saved to `study_session_logs/model_*_latest.pt`.

### 2. MONITOR REAL-TIME STATE (The "Mind's Eye")
Visualizes the AI's brain activity, loss, thoughts, and emotions in a rich terminal dashboard.

**Command:**
```powershell
python neural_monitor.py
```

*   **Features:**
    *   Live Loss Graph (12D vs 42D)
    *   Current Thought Stream
    *   Emotional State (Valence/Arousal)
    *   Current Topic

### 3. CHAT WITH THE MODEL (The "Voice")
Talk to the 12D model directly. It has memory and "synesthetic" text rendering (colors based on token meaning).

**Command:**
```powershell
$env:PYTHONPATH = "$PWD\packages\cosmic-synapse-transformer"; python production_12d/chat_console.py
```

*   **Commands:**
    *   `/reset` - Wipe conversation memory.
    *   `/temp 0.8` - Set temperature (creativity).
    *   `/exit` - Quit.

---

## 📂 File Structure

```
The-Cosmic-Davis-12D-Hebbian-Transformer--1/
├── neural_monitor.py           # Real-time dashboard
├── quiz_status.py              # Quick "1+1" intelligence test
├── test_multimodal.py          # Verify sensory systems
├── NEXT_SESSION_PROMPT.md      # Handover instructions
│
├── production_12d/             # Production-ready code
│   └── chat_console.py         # 12D Chat Interface
│
├── research_42d/               # Experimental research code
│   ├── chat_console_42d.py     # 42D Chat Interface
│   │
│   ├── audio_12d/              # Multimodal Engine
│   │   ├── audio_engine.py     # Audio -> 12D
│   │   ├── visual_engine.py    # Vision -> 12D
│   │   ├── affective_engine.py # Emotion & Thought
│   │   ├── subvocalizer.py     # Text -> Audio
│   │   └── multimodal_fusion.py# Unified System
│   │
│   └── experiments/
│       └── singularity_42d/
│           └── autonomous_study.py # MAIN TRAINING LOOP
│
├── study_session_logs/         # Logs & Checkpoints
│   ├── brain_state.json        # Live data for monitor
│   ├── study_log.txt           # Training history
│   ├── vocab.txt               # Vocabulary file
│   └── model_*.pt              # Saved models
│
└── packages/
    └── cosmic-synapse-transformer/ # Core Model Code
```

---

## 🔮 Future Roadmap

1.  **GPU Acceleration:** Porting tensor operations to CUDA for 50x speedup.
2.  **Web Interface:** React/Three.js frontend for 3D brain visualization.
3.  **Scale Up:** Increasing parameter count to 100M+ for semantic emergence ("Grokking").

---

*"The universe is not made of atoms; it is made of stories."*
