"""
COSMO'S CHAT — Interactive Console Interface
==============================================

Chat with a trained Cosmo's model via the console.

Features:
- Persistent 54D state across conversation turns
- Real-time emotional state integration (CST sensory bridge)
- Internal monologue display
- Temperature/top-k/top-p controls
- State inspection commands

Commands:
    /state       — Show current 54D internal state summary
    /metrics     — Show last generation metrics
    /temp <val>  — Set temperature
    /topk <val>  — Set top-k
    /save        — Save conversation state
    /help        — Show commands
    /quit        — Exit

Author: Cory Shane Davis
"""

import sys
import torch
from pathlib import Path
from typing import Optional

from .cosmos_config import CosmosConfig
from .cosmos_model import CosmosTransformer


def load_model(checkpoint_path: str, device: str = "auto") -> tuple:
    """Load a trained Cosmo's model from checkpoint."""
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = checkpoint["config"]

    config = CosmosConfig(
        vocab_size=cfg_dict["vocab_size"],
        d_model=cfg_dict["d_model"],
        n_layers=cfg_dict["n_layers"],
        n_heads=cfg_dict["n_heads"],
        d_state=cfg_dict.get("d_state", 54),
        max_seq_len=cfg_dict.get("max_seq_len", 2048),
        memory_size=cfg_dict.get("memory_size", 256),
        n_chaos_oscillators=cfg_dict.get("n_chaos_oscillators", 7),
    )

    model = CosmosTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(dev)
    model.eval()

    state = checkpoint.get("persistent_state")
    if state is not None:
        state = state.to(dev)

    return model, config, state, dev


def chat_loop(model, config, initial_state, device):
    """Main interactive chat loop."""
    # Setup tokenizer
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        tokenize = enc.encode
        decode = enc.decode
        print("[COSMO'S] Using tiktoken tokenizer")
    except ImportError:
        chars = list(set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,!?;:'\"-—()[]{}/@#$%^&*+=<>~`\n\t"
        ))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        tokenize = lambda text: [stoi.get(c, 0) for c in text]
        decode = lambda tokens: "".join(itos.get(t, "?") for t in tokens)
        print("[COSMO'S] Using character-level tokenizer")

    # Chat state
    state = initial_state
    temperature = 0.8
    top_k = 50
    top_p = None
    max_gen = 200
    last_metrics = {}

    # Header
    print()
    print("=" * 60)
    print("  ✨  COSMO'S — Unified 54D CST Transformer  ✨")
    print(f"  Model: {config.d_model}D, {config.n_layers}L, {config.n_heads}H")
    print(f"  State: {config.d_state}D ({config.d_state_scalar}+{config.d_state_vector})")
    print(f"  Device: {device}")
    print("=" * 60)
    print("  Type /help for commands, /quit to exit\n")

    while True:
        try:
            user_input = input("🌌 You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[COSMO'S] Goodbye!")
            break

        if not user_input:
            continue

        # ---- Commands ----
        if user_input.startswith("/"):
            cmd_parts = user_input.split()
            cmd = cmd_parts[0].lower()

            if cmd == "/quit" or cmd == "/exit":
                print("[COSMO'S] Shutting down. State preserved.")
                break

            elif cmd == "/help":
                print("Commands:")
                print("  /state       — 54D state summary")
                print("  /metrics     — Last generation metrics")
                print(f"  /temp <val>  — Set temperature (current: {temperature})")
                print(f"  /topk <val>  — Set top-k (current: {top_k})")
                print(f"  /topp <val>  — Set top-p (current: {top_p})")
                print(f"  /maxgen <n>  — Set max tokens (current: {max_gen})")
                print("  /save        — Save state to disk")
                print("  /quit        — Exit")

            elif cmd == "/state":
                if state is not None:
                    s = state
                    print(f"[54D STATE]")
                    print(f"  Shape: {s.shape}")
                    print(f"  Mean:  {s.mean().item():.6f}")
                    print(f"  Std:   {s.std().item():.6f}")
                    print(f"  Scalar (0:12): {s[:,:,:12].mean().item():.6f}")
                    print(f"  Vector (12:):  {s[:,:,12:].mean().item():.6f}")
                    print(f"  Min:   {s.min().item():.6f}")
                    print(f"  Max:   {s.max().item():.6f}")
                else:
                    print("[STATE] No persistent state yet.")

            elif cmd == "/metrics":
                if last_metrics:
                    for k, v in last_metrics.items():
                        print(f"  {k}: {v}")
                else:
                    print("[METRICS] No metrics yet.")

            elif cmd == "/temp" and len(cmd_parts) > 1:
                temperature = float(cmd_parts[1])
                print(f"[COSMO'S] Temperature set to {temperature}")

            elif cmd == "/topk" and len(cmd_parts) > 1:
                top_k = int(cmd_parts[1])
                print(f"[COSMO'S] top_k set to {top_k}")

            elif cmd == "/topp" and len(cmd_parts) > 1:
                top_p = float(cmd_parts[1])
                print(f"[COSMO'S] top_p set to {top_p}")

            elif cmd == "/maxgen" and len(cmd_parts) > 1:
                max_gen = int(cmd_parts[1])
                print(f"[COSMO'S] max_gen set to {max_gen}")

            elif cmd == "/save":
                save_path = Path("checkpoints/cosmos/chat_state.pt")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state": state, "metrics": last_metrics}, save_path)
                print(f"[COSMO'S] State saved to {save_path}")

            else:
                print(f"[COSMO'S] Unknown command: {cmd}. Type /help")

            continue

        # ---- Generate response ----
        with torch.no_grad():
            # Tokenize input
            tokens = tokenize(user_input)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)

            # Generate with state persistence
            output, new_state = model.generate(
                idx,
                max_new_tokens=max_gen,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                state_x54=state,
            )

            # Decode only the generated part
            generated_tokens = output[0, len(tokens):].tolist()
            response = decode(generated_tokens)

            # Update persistent state
            state = new_state

            # Quick metrics
            last_metrics = {
                "x54_mean": new_state.mean().item(),
                "x54_std": new_state.std().item(),
                "tokens_generated": len(generated_tokens),
            }

        print(f"\n✨ Cosmo's > {response}\n")


def main():
    """Entry point for console chat."""
    import argparse

    parser = argparse.ArgumentParser(description="Cosmo's Chat Interface")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="checkpoints/cosmos/cosmos_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, or cuda",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"[COSMO'S] No checkpoint found at {ckpt}")
        print("[COSMO'S] Starting with random weights (untrained) for testing...")
        print()

        # Create untrained model for testing
        config = CosmosConfig.tiny(vocab_size=256)
        model = CosmosTransformer(config)
        model.eval()
        dev = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
        model = model.to(dev)
        chat_loop(model, config, None, dev)
    else:
        model, config, state, dev = load_model(str(ckpt), args.device)
        chat_loop(model, config, state, dev)


if __name__ == "__main__":
    main()
