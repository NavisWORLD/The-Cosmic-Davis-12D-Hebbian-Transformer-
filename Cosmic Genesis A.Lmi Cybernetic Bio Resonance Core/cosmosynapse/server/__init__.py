"""
CosmoSynapse Server — Standalone WebSocket API

Runs the 12D CST Emotional API server on port 8765.

Usage:
    python -m cosmosynapse.server
"""

from .emotion_server import run_server

__all__ = ["run_server"]

if __name__ == "__main__":
    run_server()
