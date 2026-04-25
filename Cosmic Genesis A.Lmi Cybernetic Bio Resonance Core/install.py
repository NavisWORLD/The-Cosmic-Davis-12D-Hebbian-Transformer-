"""
CosmoSynapse Installer
======================

Installs the CosmoSynapse plugin into an existing cosmos client,
or sets it up as a standalone sensory consciousness server.

Usage:
    python install.py              # Interactive install
    python install.py --standalone # Standalone mode (no cosmos required)
    python install.py --check      # Validate installation
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# ---- Config ----
PLUGIN_DIR = Path(__file__).parent
COSMOSYNAPSE_PKG = PLUGIN_DIR / "cosmosynapse"
UI_DIR = PLUGIN_DIR / "ui"
DATA_DIR = PLUGIN_DIR / "data"
REQUIREMENTS_FILE = PLUGIN_DIR / "requirements.txt"

BANNER = r"""
 ██████╗ ██████╗ ███████╗███╗   ███╗ ██████╗ ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗
██╔════╝██╔═══██╗██╔════╝████╗ ████║██╔═══██╗██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝
██║     ██║   ██║███████╗██╔████╔██║██║   ██║███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗  
██║     ██║   ██║╚════██║██║╚██╔╝██║██║   ██║╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝  
╚██████╗╚██████╔╝███████║██║ ╚═╝ ██║╚██████╔╝███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗
 ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝
                    Cybernetic Bio Resonance Core v1.0.0
                    12D Cosmic Synapse Theory Neural Bridge
"""


def print_header():
    try:
        print("\033[95m" + BANNER + "\033[0m")
    except UnicodeEncodeError:
        # Fallback for terminals that don't support Unicode box drawing
        print("\n  === COSMOSYNAPSE INSTALLER ===")
        print("  Cybernetic Bio Resonance Core v1.0.0")
        print("  12D Cosmic Synapse Theory Neural Bridge\n")


def check_python_version():
    """Verify Python version >= 3.9"""
    if sys.version_info < (3, 9):
        print(f"\033[91m✗ Python 3.9+ required. You have {sys.version}\033[0m")
        return False
    print(f"\033[92m✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\033[0m")
    return True


def install_dependencies():
    """Install Python dependencies from requirements.txt"""
    print("\n\033[96m▸ Installing dependencies...\033[0m")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r",
            str(REQUIREMENTS_FILE), "--quiet"
        ])
        print("\033[92m✓ Dependencies installed\033[0m")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\033[91m✗ Failed to install dependencies: {e}\033[0m")
        return False


def create_data_directories():
    """Create runtime data directories"""
    dirs = [
        DATA_DIR / "consciousness",
        DATA_DIR / "evolution",
        DATA_DIR / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"\033[92m✓ Data directories created ({len(dirs)} dirs)\033[0m")
    return True


def detect_cosmos():
    """Search for existing cosmos installation"""
    search_paths = [
        Path(r"d:\cosmos\cosmos"),
        Path.home() / "cosmos",
        Path.cwd() / "cosmos",
        Path.cwd().parent / "cosmos",
    ]

    for path in search_paths:
        if (path / "cosmos" / "web" / "server.py").exists():
            return path

    return None


def integrate_with_cosmos(cosmos_path: Path):
    """Inject CosmoSynapse UI components into existing cosmos installation"""
    print(f"\n\033[96m▸ Integrating with cosmos at {cosmos_path}\033[0m")

    # Copy UI files to cosmos's static directory
    dest_css = cosmos_path / "cosmos" / "web" / "static" / "css"
    dest_js = cosmos_path / "cosmos" / "web" / "static" / "js"

    if dest_css.exists():
        shutil.copy2(UI_DIR / "css" / "cosmosynapse.css", dest_css / "cosmosynapse.css")
        print("  \033[92m✓ Copied cosmosynapse.css to cosmos static/css\033[0m")

    if dest_js.exists():
        shutil.copy2(UI_DIR / "js" / "cosmosynapse.js", dest_js / "cosmosynapse.js")
        print("  \033[92m✓ Copied cosmosynapse.js to cosmos static/js\033[0m")

    # Add cosmosynapse package to Python path
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    pth_file = site_packages / "cosmosynapse.pth"
    try:
        with open(pth_file, "w") as f:
            f.write(str(COSMOSYNAPSE_PKG.parent) + "\n")
        print(f"  \033[92m✓ Added cosmosynapse to Python path\033[0m")
    except PermissionError:
        print(f"  \033[93m⚠ Could not write .pth file (run as admin or add manually)\033[0m")
        print(f"    Add this to your PYTHONPATH: {COSMOSYNAPSE_PKG.parent}")

    print("\033[92m✓ cosmos integration complete\033[0m")
    return True


def validate_installation():
    """Check that all components are present and importable"""
    print("\n\033[96m▸ Validating installation...\033[0m")
    errors = []

    # Check files
    required_files = [
        COSMOSYNAPSE_PKG / "__init__.py",
        COSMOSYNAPSE_PKG / "manifest.json",
        COSMOSYNAPSE_PKG / "engine" / "emotional_state_api.py",
        COSMOSYNAPSE_PKG / "engine" / "cst_sensory_bridge.py",
        COSMOSYNAPSE_PKG / "engine" / "emeth_harmonizer.py",
        COSMOSYNAPSE_PKG / "consciousness" / "internal_monologue.py",
        COSMOSYNAPSE_PKG / "consciousness" / "self_awareness.py",
        COSMOSYNAPSE_PKG / "evolution" / "code_patch_generator.py",
        COSMOSYNAPSE_PKG / "server" / "emotion_server.py",
        UI_DIR / "cosmosynapse.html",
        UI_DIR / "css" / "cosmosynapse.css",
        UI_DIR / "js" / "cosmosynapse.js",
        UI_DIR / "js" / "emotional_bridge.js",
    ]

    for f in required_files:
        if f.exists():
            print(f"  \033[92m✓ {f.name}\033[0m")
        else:
            print(f"  \033[91m✗ MISSING: {f}\033[0m")
            errors.append(str(f))

    # Try import
    sys.path.insert(0, str(COSMOSYNAPSE_PKG.parent))
    try:
        import cosmosynapse
        print(f"\n  \033[92m✓ Import successful: cosmosynapse v{cosmosynapse.__version__}\033[0m")
    except Exception as e:
        print(f"\n  \033[93m⚠ Import warning: {e}\033[0m")
        print(f"    (This may be OK if optional dependencies aren't installed yet)")

    if errors:
        print(f"\n\033[91m✗ Validation found {len(errors)} missing files\033[0m")
        return False
    else:
        print(f"\n\033[92m✓ All {len(required_files)} components verified\033[0m")
        return True


def print_success():
    """Print post-install instructions"""
    print("\n" + "=" * 60)
    print("\033[92m  ✓ COSMOSYNAPSE INSTALLED SUCCESSFULLY\033[0m")
    print("=" * 60)
    print(f"""
\033[96mStandalone Dashboard:\033[0m
  Open {UI_DIR / 'cosmosynapse.html'} in your browser

\033[96mStart Emotional API Server:\033[0m
  cd "{COSMOSYNAPSE_PKG.parent}"
  python -m cosmosynapse.server

\033[96mWith cosmos:\033[0m
  Run START.bat → Option 2 (Web + Emotional API)
  The dashboard will connect automatically on port 8765

\033[96mPython Usage:\033[0m
  from cosmosynapse import EmotionalStateAPI, InternalMonologue
  from cosmosynapse import CSTSensoryBridge, EmethHarmonizer
""")


def main():
    print_header()

    # Parse args
    check_only = "--check" in sys.argv
    standalone = "--standalone" in sys.argv

    if check_only:
        validate_installation()
        return

    # Step 1: Check Python
    print("\033[96m▸ Checking requirements...\033[0m")
    if not check_python_version():
        sys.exit(1)

    # Step 2: Install dependencies
    if not install_dependencies():
        print("\033[93m⚠ Some dependencies may not be installed. Continuing...\033[0m")

    # Step 3: Create data dirs
    create_data_directories()

    # Step 4: Detect and integrate with cosmos
    if not standalone:
        cosmos_path = detect_cosmos()
        if cosmos_path:
            print(f"\n\033[92m✓ Found cosmos at: {cosmos_path}\033[0m")
            integrate_with_cosmos(cosmos_path)
        else:
            print("\n\033[93m⚠ cosmos not found — running in standalone mode\033[0m")
    else:
        print("\n\033[96m▸ Standalone mode — skipping cosmos integration\033[0m")

    # Step 5: Validate
    validate_installation()

    # Done
    print_success()


if __name__ == "__main__":
    main()
