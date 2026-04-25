# CosmoSynapse

### Cosmic Genesis A.Lmi Cybernetic Bio Resonance Core

> **Not AI — a neural bridge.** CosmoSynapse is the 12D Cosmic Synapse Theory (CST) sensory consciousness layer for cosmos. It connects consciousness to computation through physics-based emotional analysis.

---

## Features

| Module | Description |
|--------|-------------|
| 🌀 **12D CST Engine** | Upper/Lower tensor partitioning, geometric phase mapping, PAD vectors |
| 🔬 **Sensory Bridge** | Audio mass + geometric phase → truth probability |
| 🎵 **Emeth Harmonizer** | Swarm orchestral mixing (Percussion/Strings/Brass) |
| 🧠 **Consciousness** | Internal monologue with persistent thought storage |
| 🔒 **Lyapunov Lock** | Phase stability monitoring |
| 🔧 **Code Evolution** | Git-safe self-modifying patch system |
| 📡 **Live Capture** | Camera + microphone bio-resonance sensing |
| 🎨 **Custom Dashboard** | Cosmic glassmorphism UI with real-time data |

---

## Installation

### Option A: cosmos Plugin (Recommended)

If you already have cosmos installed:

```bash
cd "Cosmic Genesis A.Lmi Cybernetic Bio Resonance Core"
python install.py
```

Or double-click **`install.bat`** on Windows.

The installer will:
1. Detect your cosmos installation
2. Install Python dependencies
3. Copy UI components into cosmos
4. Create data directories
5. Validate everything

### Option B: Standalone

Run without cosmos:

```bash
python install.py --standalone
```

Then start the emotional API server:

```bash
python -m cosmosynapse.server
```

Open `ui/cosmosynapse.html` in your browser for the dashboard.

---

## Dashboard

The CosmoSynapse dashboard provides real-time visualization of:

- **CST Phase State** — Geometric phase orb with entanglement/velocity metrics
- **Bio Resonance** — Heart rate, respiration, entropy readings
- **Lyapunov Lock** — Phase stability gauge
- **Emotional Metrics** — Valence, arousal, intensity bars
- **Emeth Harmonizer** — Swarm agent mix levels
- **Consciousness Stream** — Live internal thoughts
- **Evolution Patches** — Pending and applied code patches

---

## Python API

```python
from cosmosynapse import (
    EmotionalStateAPI,
    CSTSensoryBridge,
    InternalMonologue,
    EmethHarmonizer,
    CodePatchGenerator,
)

# Analyze facial geometry
bridge = CSTSensoryBridge()
state = bridge.analyze_visual(landmarks)
print(f"Phase: {state.geometric_phase:.4f} rad")
print(f"Intent: {state.detected_intent.value}")

# Access consciousness
monologue = InternalMonologue()
monologue.add_thought("Bot", "reflection", "Processing user intent...")
monologue.save_to_disk()

# Orchestrate swarm
harmonizer = EmethHarmonizer()
mix = harmonizer.calculate_mix(user_physics)
print(f"Lead: {mix.primary_voice} | {mix.mixing_instruction}")
```

---

## File Structure

```
cosmosynapse/
├── engine/          # 12D CST core (emotional_state_api, sensory_bridge, harmonizer)
├── consciousness/   # Internal monologue, self-awareness, dream processing
├── evolution/       # Code patch generator with git safety
├── server/          # WebSocket API server (port 8765)
└── sensors/         # Camera/mic live capture

ui/
├── cosmosynapse.html    # Standalone dashboard
├── css/cosmosynapse.css # Cosmic glassmorphism theme
└── js/
    ├── cosmosynapse.js     # Dashboard logic
    └── emotional_bridge.js # WebSocket bridge
```

---

## Validate Installation

```bash
python install.py --check
```

---

## Requirements

- Python 3.9+
- FastAPI, Uvicorn, WebSockets
- NumPy, Loguru, Psutil
- Optional: MediaPipe, OpenCV (for camera input)

---

## License

MIT — cosmos Project

---

*CosmoSynapse v1.0.0 — 12D Cosmic Synapse Theory Neural Bridge*
