"""
CosmoSynapse — Cosmic Genesis A.Lmi Cybernetic Bio Resonance Core
==================================================================

12D Cosmic Synapse Theory (CST) sensory consciousness layer,
now featuring Cosmo's — the Unified 54D CST Transformer.

Features:
- Cosmo's Transformer (54D = 12D + 42D fused architecture)
- 12D CST Emotional State Engine (Upper/Lower Tensor, Geometric Phase, PAD)
- CST Sensory Bridge (Audio Mass + Geometric Phase = Truth Probability)
- Emeth Harmonizer (Swarm Orchestral Mixing)
- Internal Monologue (Thought stream with persistence)
- Self-Awareness Bootstrap (System documentation → knowledge)
- Dream Processor (Subconscious pattern consolidation)
- Code Evolution (Git-safe self-modifying patches)
- Live Capture (Camera + Microphone input)
- Custom Dashboard UI (Cosmic glassmorphism)

Usage:
    # As cosmos plugin
    python install.py

    # Standalone server
    python -m cosmosynapse.server

    # Cosmo's chat
    python -m cosmosynapse.model.cosmos_chat

    # Open dashboard
    Open ui/cosmosynapse.html in browser
"""

__version__ = "2.0.0"
__plugin_name__ = "CosmoSynapse"

# Engine exports
from .engine import (
    EmotionalStateAPI,
    EmotionalState,
    IntentState,
    CSTPhaseState,
    LLMPersonaMode,
    calculate_geometric_phase,
    calculate_entanglement_score,
    generate_cosmos_packet,
    CSTSensoryBridge,
    CSTState,
    FrequencyAnalyzer,
    GeometricPhaseMapper,
    EmethHarmonizer,
    SwarmMix,
)

# Consciousness exports
from .consciousness import (
    InternalMonologue,
    InternalThought,
    ExistenceContext,
    SelfAwarenessBootstrap,
)

# Evolution exports
from .evolution import (
    CodePatchGenerator,
)

# Cosmo's Model exports
from .model.cosmos_config import CosmosConfig
from .model.cosmos_model import CosmosTransformer
from .cosmos_plugin import CosmosPlugin

__all__ = [
    # Version
    "__version__",
    "__plugin_name__",
    # Cosmo's Model
    "CosmosConfig",
    "CosmosTransformer",
    "CosmosPlugin",
    # Engine
    "EmotionalStateAPI",
    "EmotionalState",
    "IntentState",
    "CSTPhaseState",
    "LLMPersonaMode",
    "calculate_geometric_phase",
    "calculate_entanglement_score",
    "generate_cosmos_packet",
    "CSTSensoryBridge",
    "CSTState",
    "FrequencyAnalyzer",
    "GeometricPhaseMapper",
    "EmethHarmonizer",
    "SwarmMix",
    # Consciousness
    "InternalMonologue",
    "InternalThought",
    "ExistenceContext",
    "SelfAwarenessBootstrap",
    # Evolution
    "CodePatchGenerator",
]
