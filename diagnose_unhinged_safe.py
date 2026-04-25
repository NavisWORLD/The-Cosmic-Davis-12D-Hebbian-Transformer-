import subprocess
import os
from pathlib import Path

ROOT_DIR = Path(os.getcwd())
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT_DIR / "packages" / "cosmic-synapse-transformer")

print("Attempting to run cosmic_unhinged_hybrid.py...")
try:
    # Run and capture output
    process = subprocess.Popen(
        ["python", "-u", "cosmic_unhinged_hybrid.py"], # -u for unbuffered
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        stdin=subprocess.PIPE
    )
    
    try:
        stdout, stderr = process.communicate(timeout=20)
    except subprocess.TimeoutExpired as e:
        process.kill()
        stdout, stderr = process.communicate()
        # In some versions of Python, communicate() might return None here
        # or e.stdout might be available.
        if stdout is None: stdout = ""
        if stderr is None: stderr = ""
        print("Process timed out.")
    
    print("\n--- STDOUT ---")
    print(stdout)
    print("\n--- STDERR ---")
    print(stderr)
    
    with open("diagnostic_results_safe.txt", "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(stdout)
        f.write("\n\nSTDERR:\n")
        f.write(stderr)

except Exception as e:
    print(f"Failed to run diagnosis: {e}")
