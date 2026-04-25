#!/usr/bin/env python3
"""
Run Farnsworth Web Interface + Emotional API Server

Starts both:
1. Web Chat Interface on port 8081
2. Emotional Token Server on port 8765

Usage:
    python run_web.py
    python run_web.py --port 8081 --host 0.0.0.0
"""

import argparse
import os
import sys
import threading
import time

# IMPORT TORCH EARLY TO PREVENT DLL CONFLICTS (shm.dll error)
try:
    import torch
except ImportError:
    pass

# FORCE LOCAL IMPORT PRIORITY
# Add critical debugging to confirm path and loaded module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
print(f"DEBUG: RunWeb CWD: {os.getcwd()}")
print(f"DEBUG: RunWeb Dir: {current_dir}")

try:
    import cosmos.web.server
    print(f"DEBUG: Loaded Server Module: {farnsworth.web.server.__file__}")
except ImportError as e:
    print(f"DEBUG: Server Import Failed: {e}")

def start_emotional_server(host="0.0.0.0", port=8765):
    """Start the emotional API server in a background thread."""
    try:
        # Add emotional_api to path
        emotional_api_path = os.path.join(os.path.dirname(__file__), "emotional_api")
        sys.path.insert(0, emotional_api_path)
        
        from emotion_server import run_server
        run_server(host=host, port=port)
    except ImportError as e:
        print(f"⚠️  Emotional API not available: {e}")
    except Exception as e:
        print(f"⚠️  Emotional API server error: {e}")

def main():
    # Get default port from env or use 8081 (avoiding conflict with Apache on 8080)
    default_port = int(os.environ.get("FARNSWORTH_WEB_PORT", "8081"))
    
    parser = argparse.ArgumentParser(description="Farnsworth Web Interface + Emotional API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=default_port, help="Web interface port")
    parser.add_argument("--emotion-port", type=int, default=8765, help="Emotional API port")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (no token verification)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--no-emotion", action="store_true", help="Disable emotional API server")
    args = parser.parse_args()

    # Set environment variables
    os.environ["FARNSWORTH_WEB_HOST"] = args.host
    os.environ["FARNSWORTH_WEB_PORT"] = str(args.port)

    if args.demo:
        os.environ["FARNSWORTH_DEMO_MODE"] = "true"

    # Import uvicorn
    import uvicorn

    print(f"""
    +==================================================================+
    |             * Farnsworth Neural Interface v2.8                  |
    |                  + 12D CST Emotional Engine                      |
    +==================================================================+
    |  Web Interface:    http://localhost:{args.port:<5}                          |
    |  Emotional API:    http://localhost:{args.emotion_port:<5}                          |
    |  Demo Mode:        {'Yes' if args.demo or os.getenv('FARNSWORTH_DEMO_MODE', 'true').lower() == 'true' else 'No ':<4}                                          |
    +==================================================================+
    |  Emotional Endpoints:                                            |
    |    GET  /state         - Current farnsworth_packet               |
    |    GET  /stream        - SSE token stream                        |
    |    WS   /ws            - WebSocket tokens                        |
    |    GET  /system_prompt - LLM steering prompt                     |
    +==================================================================+
    |  Required Token: 9crfy4udr...wBAGS                               |
    +==================================================================+
    """)

    # Check if emotional port is already in use
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', args.emotion_port))
    is_port_open = result == 0
    sock.close()

    if is_port_open:
        print(f"    [WARN]  Port {args.emotion_port} is busy - Emotional API probably running externally. Skipping internal start.")
        args.no_emotion = True

    # -------------------------------------------------------------
    # QUANTUM BRIDGE CONNECTION TEST
    # -------------------------------------------------------------
    print("    [INIT]  Testing Quantum Bridge Connection...")
    try:
        from cosmos.core.quantum_bridge import get_quantum_bridge
        qb = get_quantum_bridge()
        if qb and qb.connect():
             entropy = qb.get_entropy()
             print(f"    [OK]    Quantum Bridge Active | Entropy Source: {qb.backend_name} | Value: {entropy:.4f}")
        else:
             print("    [WARN]  Quantum Bridge Offline - System will run in DETERMINISTIC mode.")
    except Exception as e:
        print(f"    [FAIL]  Quantum Connection Error: {e}")
    # -------------------------------------------------------------

    # Start emotional API server in background thread (unless disabled)
    if not args.no_emotion:
        emotion_thread = threading.Thread(
            target=start_emotional_server,
            args=(args.host, args.emotion_port),
            daemon=True
        )
        emotion_thread.start()
        print(f"    🎭 Emotional API Server starting on port {args.emotion_port}...")
        time.sleep(1)  # Give it time to start
    
    # Start main web server
    uvicorn.run(
        "cosmos.web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
