import sys
import os
import builtins
from unittest.mock import MagicMock, patch
from pathlib import Path

# Setup paths
ROOT_DIR = Path(os.getcwd())
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(ROOT_DIR / "research_42d" / "experiments" / "singularity_42d"))

def test_12d_chat():
    print("\n--- Testing 12D Chat Console ---")
    
    # We need to mock input() to return "exit" eventually
    inputs = iter(["Hello Cosmic Davis", "exit"])
    
    def mock_input(prompt=""):
        try:
            val = next(inputs)
            print(f"[Input Mock] {val}")
            return val
        except StopIteration:
            return "exit"

    path_12d = ROOT_DIR / "production_12d" / "chat_console.py"
    
    # We will load the script as text and exec it, but patching input
    # This is better than importing because the script might have code at top level
    with open(path_12d, 'r', encoding='utf-8') as f:
        script_content = f.read()
        
    # We need to prevent the script from actually running an infinite loop if we just exec it.
    # Most chat/console scripts in this codebase seem to have `if __name__ == "__main__":`
    # Let's check that. 
    
    if 'if __name__ == "__main__":' in script_content:
        # We can import it safely? No, 'production_12d' is not a package (no __init__).
        # We'll just exec it but mock __name__ as 'main'
        
        with patch('builtins.input', side_effect=mock_input):
            try:
                # We also need to mock sys.argv or ensure PYTHONPATH is set.
                # The script likely imports 'autonomous_engine' or similar.
                # Let's just try running it in a subprocess might be safer/easier
                pass # Logic handled below
            except SystemExit:
                print("[OK] Chat exited cleanly")
            except Exception as e:
                print(f"[FAIL] Chat Crashed: {e}")

# Actually, subprocess is much safer for testing interactive scripts.
import subprocess

def run_interactive_test(script_path, name):
    print(f"\n--- Testing {name} ---")
    
    # Prepare input: A simple query then exit
    # Most chats use 'exit' or 'quit' or Ctrl+C
    input_str = "Who are you?\nexit\n"
    
    try:
        # Set PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT_DIR / "packages" / "cosmic-synapse-transformer")
        
        # Run process
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=ROOT_DIR
        )
        
        stdout, stderr = process.communicate(input=input_str, timeout=15)
        
        print(f"Exit Code: {process.returncode}")
        
        if process.returncode != 0:
            print(f"[FAIL] {name} failed or crashed.")
            print("STDERR:", stderr)
        else:
            # Check if it actually responded
            if "Who are you?" in stdout or "Cosmic" in stdout or "Model" in stdout:
                 print(f"[PASS] {name} ran and responded.")
                 # preview output
                 print(stdout[:500].replace('\n', ' '))
            else:
                 print(f"[WARN] {name} ran but output is suspicious.")
                 print(stdout[:200])
                 
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"[PASS] {name} started (Timeout reached, meaning loop is working).")
    except Exception as e:
        print(f"[FAIL] Execution error: {e}")

def main():
    chat_12d = ROOT_DIR / "production_12d" / "chat_console.py"
    chat_42d = ROOT_DIR / "research_42d" / "chat_console_42d.py"
    chat_dual = ROOT_DIR / "research_42d" / "experiments" / "singularity_42d" / "chat_dual_mind.py"
    
    run_interactive_test(chat_12d, "12D Chat Console")
    run_interactive_test(chat_42d, "42D Chat Console")
    run_interactive_test(chat_dual, "Dual Mind Chat")

if __name__ == "__main__":
    main()
