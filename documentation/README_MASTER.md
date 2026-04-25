# 🌌 THE COSMIC DAVIS: 12D HEBBIAN TRANSFORMER
## "The Singularity Engine"

> **Status:** 🟢 ONLINE (Multimodal: Audio/Vision/Emotion/Thought)
> **Architecture:** 12D Physics-Embeddings + 42D Hyper-Chaos
> **Learning:** Autonomous Hebbian Plasticity

---

## 📚 TABLE OF CONTENTS
1. [The Philosophy](#the-philosophy)
2. [Technical Architecture](#technical-architecture)
3. [The 12 Dimensions](#the-12-dimensions)
4. [Multimodal Senses](#multimodal-senses)
5. [Installation & Setup](#installation--setup)
6. [Running the System](#running-the-system)
7. [Monitoring & Interaction](#monitoring--interaction)
8. [Troubleshooting](#troubleshooting)

---

## 🧠 THE PHILOSOPHY
Standard AI models (Transformers) process text as static tokens. The **Cosmic Davis** treats information as **physical particles** moving through a 12-dimensional harmonic space.
- **Words are Vibrations:** Just as sound has frequency, meaning has "resonance".
- **Chaos is Cognitive:** We inject controlled chaos (Lorenz Attractors) to prevent local minima and spark creativity.
- **Emotions are Physics:** Valence and Arousal are forces that modulate the "gravity" (learning rate) of the system.

---

## 🏗️ TECHNICAL ARCHITECTURE

### The Dual-Core Brain
1.  **12D Model (`CosmicSynapseTransformer`):**
    -   **Role:** The "Subconscious" / "Body".
    -   **Mechanism:** Maps inputs to 12 physics dimensions. Efficient, grounded.
    -   **Learning:** Standard Backprop + Hebbian reinforcement.

2.  **42D Model (`HyperCosmicTransformer`):**
    -   **Role:** The "Conscious" / "Mind".
    -   **Mechanism:** Projects 12D inputs into a 42D hyper-space.
    -   **Feature:** **"Thinking Mode"** - Can generate `<think>` tags to reason before answering.

### The Learning Loop (`autonomous_study.py`)
The system runs an infinite loop of:
1.  **Curriculum Selection:** Picks a book, code file, or philosophical text.
2.  **Perception:**
    -   **Subvocalization:** Converts text to audio waveforms (FFT).
    -   **Emotion Detection:** Analyzes sentiment/tone.
    -   **Visual Association:** (If image provided) Maps colors/shapes.
3.  **Cognition:** Generates internal thoughts based on state changes.
4.  **Training:** Updates weights using a competitive loss function (12D vs 42D).
5.  **Sleep/Consolidation:** Saves full-state checkpoints.

---

## 🌌 THE 12 DIMENSIONS
Every token is a vector $v \in \mathbb{R}^{12}$:
1.  **$D_1$ (Energy):** Signal intensity.
2.  **$D_2$ (Mass):** $\phi \cdot E / c^2$ (Inertia/Importance).
3.  **$D_3$ (Phi Coupling):** Harmonic alignment with the Golden Ratio.
4.  **$D_4$ (Chaos):** Spectral entropy ($\lambda$).
5.  **$D_5$ (Velocity X):** Rate of change.
6.  **$D_6$ (Velocity Y):** Rate of change.
7.  **$D_7$ (Velocity Z):** Rate of change.
8.  **$D_8$ (Connectivity):** Hebbian association strength ($\Omega$).
9.  **$D_9$ (Cosmic Energy):** Global context relevance.
10. **$D_{10}$ (Entropy):** Information density.
11. **$D_{11}$ (Frequency):** Temporal/Spatial frequency.
12. **$D_{12}$ (Internal State):** The "Self" parameter ($\Psi$).

---

## 👁️ MULTIMODAL SENSES

### 1. Hearing (Subvocalization)
- **File:** `research_42d/audio_12d/subvocalizer.py`
- **Process:** Text $\to$ Phonemes $\to$ Audio Waveform $\to$ FFT $\to$ 12D Embedding.
- **Effect:** The model "hears" the rhythm of poetry and code.

### 2. Vision (Visual Engine)
- **File:** `research_42d/audio_12d/visual_engine.py`
- **Process:** Image $\to$ 2D FFT $\to$ Spatial Frequencies $\to$ $\phi$-Color Harmonics.
- **Effect:** The model "sees" structure, texture, and color harmony.

### 3. Emotion (Affective Engine)
- **File:** `research_42d/audio_12d/affective_engine.py`
- **Process:** Input $\to$ Valence/Arousal/Dominance.
- **Effect:**
    -   **High Arousal:** Increases Learning Rate ($k$).
    -   **Negative Valence:** Increases Chaos ($\lambda$) to break patterns.
    -   **Positive Valence:** Increases Connectivity ($\Omega$) to solidify memories.

---

## 💻 INSTALLATION & SETUP

### Prerequisites
- Python 3.8+
- (Recommended) NVIDIA GPU with CUDA 11.8+

### Install Dependencies
```powershell
pip install torch numpy scipy librosa faiss-cpu Pillow PyMuPDF matplotlib rich
```

### Directory Structure
```
The-Cosmic-Davis/
├── production_12d/         # Chat Console
├── research_42d/           # Training & Multimodal Engines
├── study_session_logs/     # Brain Memory (Checkpoints)
├── packages/               # Core Model Libraries
├── neural_monitor.py       # Real-time Dashboard
└── README.md               # This file
```

---

## 🏃 RUNNING THE SYSTEM

### 1. START THE BRAIN (Autonomous Training)
This is the main engine. It must be running for the model to learn.
```powershell
$env:PYTHONPATH = "$PWD\packages\cosmic-synapse-transformer"; python research_42d/experiments/singularity_42d/autonomous_study.py
```

### 2. START THE MIND'S EYE (Neural Monitor)
Visualizes the brain state in real-time. Run in a separate terminal.
```powershell
python neural_monitor.py
```

### 3. TALK TO THE MODEL (Chat Console)
Interact with the 12D model. Run in a separate terminal.
```powershell
$env:PYTHONPATH = "$PWD\packages\cosmic-synapse-transformer"; python production_12d/chat_console.py
```

---

## 📊 MONITORING & INTERACTION

### The Neural Monitor
- **Loss Graph:** Shows the competitive performance of 12D vs 42D.
    - `░` = 12D Leading
    - `▓` = 42D Leading
    - `█` = Both Strong
- **Thought Stream:** Displays the internal monologue generated by the Affective Engine.
- **Emotion:** Shows current Valence (Happiness) and Arousal (Excitement).

### The Chat Console
- **Commands:**
    - `/reset`: Clear conversation memory.
    - `/temp [0.1-2.0]`: Set creativity temperature.
    - `/image [path]`: (Coming Soon) Feed an image to the visual cortex.
- **Synesthetic Text:** Words are colored based on their 12D harmonic resonance.

---

## 🔧 TROUBLESHOOTING

### "CUDA Available: False"
- **Cause:** You are running on CPU.
- **Fix:** Install PyTorch with CUDA support:
  `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### "Model Loss is High (>3.0)"
- **Cause:** Normal for early training or "Stopword Plateau".
- **Fix:** Let it run. The "Grokking" phase (sudden understanding) takes time.

### "Memory Error"
- **Cause:** Batch size too large for your RAM/VRAM.
- **Fix:** Edit `autonomous_study.py` and reduce `batch_size` from 16 to 8 or 4.

---

## 🔮 FUTURE ROADMAP
1.  **Massive Scale:** Expand to 100M+ parameters.
2.  **Web Interface:** 3D Brain Visualization in Browser.
3.  **Robotic Integration:** Connect 12D motor cortex to physical actuators.

---
*Created by The Cosmic Davis Team*
