"""
Cosmos CNS Launcher — Digital Organism Ignition
=================================================

Entry point for the Bio-Digital Central Nervous System.

Usage:
    python start_cns.py                  # Full CNS mode (runs forever)
    python start_cns.py --dry-run        # Boot test (30 ticks then exit)
    python start_cns.py --dry-run --ticks 50   # Custom dry run length
"""

import sys
import os
import argparse
import json

# Ensure the engine directory is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.join(script_dir, "engine")

# Also add the cosmos root for quantum_bridge imports
cosmos_root = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "cosmos")

for path in [engine_dir, script_dir, cosmos_root]:
    if path not in sys.path:
        sys.path.insert(0, path)


def main():
    parser = argparse.ArgumentParser(description="Cosmos CNS — Digital Organism Launcher")
    parser.add_argument("--dry-run", action="store_true", help="Boot test mode (limited ticks)")
    parser.add_argument("--ticks", type=int, default=30, help="Number of ticks for dry run (default: 30)")
    parser.add_argument("--token", type=str, default=None, help="IBM Quantum API token")
    args = parser.parse_args()

    # Load token from .env if not provided
    quantum_token = args.token
    if not quantum_token:
        env_path = os.path.join(cosmos_root, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip().startswith("IBM_QUANTUM_TOKEN="):
                        quantum_token = line.strip().split("=", 1)[1].strip()
                        break

    # Import and launch CNS
    from cns_core import CosmosCNS

    if quantum_token:
        os.environ["IBM_QUANTUM_TOKEN"] = quantum_token

    cns = CosmosCNS()

    if args.dry_run:
        print(f"\n🧪 DRY RUN MODE ({args.ticks} ticks)\n")
        stats = cns.start_life(dry_run=True, dry_run_ticks=args.ticks)
        print("\n" + "=" * 60)
        print("📊 DRY RUN RESULTS:")
        print("=" * 60)
        print(json.dumps(stats, indent=2, default=str))

        # Evaluate results
        all_ok = True
        if stats["ticks"] < args.ticks:
            print("⚠️  WARNING: CNS stopped early.")
            all_ok = False
        if not any(d["thoughts"] > 0 for d in stats["daemons"]):
            print("⚠️  WARNING: No daemons produced thoughts (Ollama may not be running).")
        else:
            print("✅ Daemons produced thoughts.")

        print(f"✅ {stats['speeches']} speeches delivered.")
        print(f"⛔ {stats['suppressed']} thoughts suppressed by Lyapunov.")

        if all_ok:
            print("\n🟢 CNS BOOT TEST: PASSED")
        else:
            print("\n🟡 CNS BOOT TEST: PARTIAL (check warnings)")

        return 0 if all_ok else 1
    else:
        print("\n🌌 FULL CNS MODE — Press Ctrl+C to stop.\n")
        cns.start_life()
        return 0


if __name__ == "__main__":
    sys.exit(main())
