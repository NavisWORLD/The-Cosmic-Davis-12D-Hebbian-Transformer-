import subprocess
import os
from pathlib import Path

ROOT_DIR = Path(os.getcwd())
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT_DIR / "packages" / "cosmic-synapse-transformer")

print("Attempting to run cosmic_unhinged_hybrid.py...")
try:
    # Run for 2 seconds and terminate
    process = subprocess.Popen(
        ["python", "cosmic_unhinged_hybrid.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        stdin=subprocess.PIPE
    )
    
    try:
        stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        print("Process timed out (expected if it's waiting for input)")
    
    print("\n--- STDOUT ---")
    print(stdout)
    print("\n--- STDERR ---")
    print(stderr)
    
    with open("diagnostic_results.txt", "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(stdout)
        f.write("\n\nSTDERR:\n")
        f.write(stderr)

except Exception as e:
    print(f"Failed to run diagnosis: {e}")
