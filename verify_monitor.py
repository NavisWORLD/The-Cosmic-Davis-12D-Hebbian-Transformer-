import sys
import os
import json
import time
from pathlib import Path
from rich.console import Console

# --- SETUP PATHS ---
ROOT_DIR = Path(os.getcwd())
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(ROOT_DIR))

def verify_monitor():
    print("="*60)
    print("🖥️  NEURAL DASHBOARD VERIFICATION")
    print("="*60)
    
    # 1. State File Check
    print("[TEST] Checking Brain State Communication...")
    log_dir = ROOT_DIR / "study_session_logs"
    state_file = log_dir / "brain_state.json"
    
    # Ensure dir exists
    log_dir.mkdir(exist_ok=True)
    
    # Create valid dummy data
    dummy_state = {
        "iteration": 999,
        "topic": "TEST_MODE",
        "loss_12d": 0.5,
        "loss_42d": 0.4,
        "emotion": "Curiosity",
        "valence": 0.8,
        "arousal": 0.6,
        "current_thought": "Verifying visual cortex...",
        "last_tokens": ["hello", "world", "<freq_5>"],
        "audio_token_count": 12,
        "winner": "42D"
    }
    
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(dummy_state, f)
        print("   [OK] Wrote dummy brain state.")
    except Exception as e:
        print(f"   [FAIL] Could not write state file: {e}")
        return

    # 2. UI Render Check (Headless)
    print("\n[TEST] Initializing UI Components...")
    try:
        from app.ui.monitor import NeuralMonitor
        
        # Instantiate
        monitor = NeuralMonitor()
        
        # Load state
        state = monitor.load_state()
        if state and state['topic'] == "TEST_MODE":
             print("   [OK] Monitor successfully read brain state.")
        else:
             print("   [FAIL] Monitor failed to read brain state.")
             
        # Mock Layout Generation (Don't run .run() loop effectively)
        layout = monitor.make_layout()
        print("   [OK] Layout Assembly (Rich) successful.")
        
        # Verify Specific Panels
        print("\n[TEST] Panel Verification:")
        monitor.update_buffers(dummy_state)
        
        # Audio Metrics
        monitor.audio_metrics_panel(dummy_state)
        print("   [OK] Audio Metrics Panel")
        
        # Token Stream
        monitor.token_panel(dummy_state)
        print("   [OK] Token Stream Panel")
        
        # Matrix
        monitor.matrix_panel(dummy_state)
        print("   [OK] 12D Matrix Panel")
        
        print("\n[SUCCESS] Neural Monitor is ready to launch.")
        print("          (Use Option [7] in Launcher to see it live)")
        
    except ImportError as e:
         print(f"   [FAIL] Import Error: {e}")
         print("          (Ensure 'rich' library is installed)")
    except Exception as e:
         print(f"   [FAIL] UI Crash: {e}")

if __name__ == "__main__":
    verify_monitor()
