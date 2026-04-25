"""
CosmoSynapse Sensors — Live Capture Module

Camera and microphone input for real-time emotional analysis.
"""

try:
    from .live_capture import LiveCapture, check_devices
except ImportError:
    LiveCapture = None
    check_devices = None

__all__ = ["LiveCapture", "check_devices"]
