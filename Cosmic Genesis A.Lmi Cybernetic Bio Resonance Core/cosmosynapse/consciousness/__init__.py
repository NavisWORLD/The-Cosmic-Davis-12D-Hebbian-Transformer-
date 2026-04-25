"""
CosmoSynapse Consciousness — Internal Awareness Layer

- InternalMonologue: Thought stream with JSON persistence
- SelfAwarenessBootstrap: System documentation → knowledge
- DreamProcessor: Subconscious pattern consolidation
"""

from .internal_monologue import (
    InternalMonologue,
    InternalThought,
    ExistenceContext,
)

from .self_awareness import (
    SelfAwarenessBootstrap,
    awaken,
    awaken_sync,
)

try:
    from .dream_processor import DreamProcessor
except ImportError:
    DreamProcessor = None

__all__ = [
    "InternalMonologue",
    "InternalThought",
    "ExistenceContext",
    "SelfAwarenessBootstrap",
    "awaken",
    "awaken_sync",
    "DreamProcessor",
]
