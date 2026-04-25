"""
AUTONOMOUS DUAL‑MIND STUDY SESSION
===================================
This script runs a continuous, self‑directed learning loop for the 12D and 42D models.
It integrates:
- Text curriculum (Gutenberg, raw README files, etc.)
- Real‑time microphone audio converted to frequency tokens (via RealTimeAudioPipe)
- Multimodal affective processing (emotion, thought)
- Live brain‑state broadcasting for the Neural Monitor.
"""

import sys
import os
import time
import random
import datetime
import json
import subprocess
import platform
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Project‑specific imports – adjust PYTHONPATH as needed before running
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(ROOT_DIR / "app" / "modules" / "dual_mind")) # For hyper_cosmic_model
sys.path.append(str(ROOT_DIR / "app" / "modules" / "audio"))

from cosmic_synapse.models.cosmic_synapse_transformer import CosmicSynapseTransformer, CosmicConfig
from cosmic_synapse.data.web_reader import WebCurriculum
from cosmic_synapse.training.unified_trainer import UnifiedCosmicTrainer
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from hyper_cosmic_model import HyperCosmicTransformer, HyperConfig
from multimodal_fusion import UnifiedMultimodalSystem
from real_time_audio_pipe import RealTimeAudioPipe

# ---------------------------------------------------------------------------
# Curriculum definition
# ---------------------------------------------------------------------------
TOPICS = {
    "LITERATURE": [
        "https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
        "https://www.gutenberg.org/cache/epub/84/pg84.txt",  # Frankenstein
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
    ],
    "PHILOSOPHY": [
        "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",  # The Republic (Plato)
        "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",  # Beyond Good and Evil (Nietzsche)
        "https://www.gutenberg.org/cache/epub/74/pg74.txt",  # Tom Sawyer (Mark Twain) – wisdom proxy
    ],
    "SCIENCE": [
        "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",  # Origin of Species (Darwin)
        "https://www.gutenberg.org/cache/epub/30155/pg30155.txt",  # Relativity (Einstein)
    ],
    "HISTORY": [
        "https://www.gutenberg.org/cache/epub/5/pg5.txt",  # US Constitution
        "https://www.gutenberg.org/cache/epub/1080/pg1080.txt",  # A Modest Proposal
    ],
    "TECHNOLOGY": [
        "https://raw.githubusercontent.com/python/cpython/main/README.rst",  # Python README
        "https://raw.githubusercontent.com/torvalds/linux/master/README",  # Linux README
        "https://www.gutenberg.org/cache/epub/3600/pg3600.txt",  # The Art of War (Strategy)
    ],
    "LIVE_WORLD": [
        "https://text.npr.org/", # Lightweight text news
        "https://lite.cnn.com", # Text-heavy news
    ],
    "SELF_AWARENESS": [
        "LOCAL_SYSTEM_SCAN", # Special token for generating local data
    ]
}

# ---------------------------------------------------------------------------
def setup_logging() -> Path:
    """Create (or reuse) a directory for all log artefacts."""
    log_dir = ROOT_DIR / "study_session_logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir

def log_progress(log_file: Path, message: str) -> None:
    """Write a timestamped line to both stdout and the log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

def get_local_system_data():
    """Generate a text report of the local system status for grounding."""
    report = []
    report.append(f"SYSTEM_TIMESTAMP: {datetime.datetime.now().isoformat()}")
    report.append(f"OS_PLATFORM: {platform.system()} {platform.release()}")
    report.append(f"MACHINE_NODE: {platform.node()}")
    report.append(f"PROCESSOR: {platform.processor()}")
    
    # Run harmless system commands to get network/env context
    try:
        # IP Configuration (Network Awareness)
        if platform.system() == "Windows":
            cmd = "ipconfig"
        else:
            cmd = "ifconfig"
        
        output = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
        report.append(f"NETWORK_CONFIG_SCAN:\n{output[:500]}...") # Limit length
    except:
        pass

    return "\n".join(report)

# ---------------------------------------------------------------------------
# Main autonomous study routine
# ---------------------------------------------------------------------------
def run_autonomous_study():
    log_dir = setup_logging()
    log_file = log_dir / "study_log.txt"

    # -------------------------------------------------------------------
    # Header & banner
    # -------------------------------------------------------------------
    log_progress(log_file, "=" * 60)
    log_progress(log_file, "STARTING AUTONOMOUS DUAL‑MIND STUDY SESSION")
    log_progress(log_file, "Curriculum: Literature, Philosophy, Science, History, Technology")
    log_progress(log_file, "=" * 60)
    print("\n# ≡ƒîî START DUAL MIND TRAINING (Evolution System)\n#   - 12D + 42D debate and collaborate\n#   - Infinite memory checkpoints\n")

    # -------------------------------------------------------------------
    # 1️⃣ Build vocabulary (quick pre‑scan of a few representative books)
    # -------------------------------------------------------------------
    gen = SyntheticDataGenerator()
    vocab_path = log_dir / "vocab.txt"
    if vocab_path.exists():
        log_progress(log_file, "Loading existing vocabulary (will be rebuilt for safety)")
    log_progress(log_file, "BUILDING FULL VOCABULARY (pre‑scanning a subset of books)…")
    sample_urls = [
        TOPICS["LITERATURE"][0],
        TOPICS["SCIENCE"][0],
        TOPICS["PHILOSOPHY"][0],
        TOPICS["HISTORY"][0],
    ]
    all_samples = []
    for url in sample_urls:
        log_progress(log_file, f"Scanning for vocab: {url}")
        try:
            reader = WebCurriculum([url])
            for _, content in reader.read_stream():
                all_samples.append(content)
        except Exception as e:
            log_progress(log_file, f"Failed to scan {url}: {e}")
    full_text = " ".join(all_samples)
    tokens = full_text.split()
    gen.build_vocabulary(tokens, max_vocab_size=30000)
    vocab_size = gen.vocab_size
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word, idx in gen.vocab.items():
            f.write(f"{word}\t{idx}\n")
    log_progress(log_file, f"FULL VOCABULARY INITIALIZED: {vocab_size} tokens")
    log_progress(log_file, f"Vocabulary saved to {vocab_path}")

    # -------------------------------------------------------------------
    # 2️⃣ Initialise models (256‑dim, 6‑layer, 8‑head)
    # -------------------------------------------------------------------
    config_12d = CosmicConfig(vocab_size=vocab_size, d_model=256, n_layers=6, n_heads=8)
    model_12d = CosmicSynapseTransformer(config_12d)
    trainer_12d = UnifiedCosmicTrainer(model_12d, config_12d, model_type="12D")

    config_42d = HyperConfig(vocab_size=vocab_size, d_model=256, n_layers=6, n_heads=8)
    model_42d = HyperCosmicTransformer(config_42d)
    trainer_42d = UnifiedCosmicTrainer(model_42d, config_42d, model_type="42D")

    log_progress(log_file, "Models initialized (256 dim, 6 layers, 8 heads)")

    # -------------------------------------------------------------------
    # 3️⃣ Initialise multimodal system and real‑time audio pipe
    # -------------------------------------------------------------------
    log_progress(log_file, "Initializing Multimodal Sensory System…")
    senses = UnifiedMultimodalSystem()
    audio_pipe = RealTimeAudioPipe()
    audio_pipe.start()
    log_progress(log_file, "Real‑time audio pipe started.")

    base_lr = 3e-4 * 0.618  # phi‑scaled learning rate
    iteration = 0

    # Directory for historic checkpoints (infinite memory)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    try:
        while True:
            iteration += 1
            # -----------------------------------------------------------
            # Choose a random topic and fetch its text
            # -----------------------------------------------------------
            topic = random.choice(list(TOPICS.keys()))
            url_source = random.choice(TOPICS[topic])
            
            log_progress(log_file, f"\n--- SESSION {iteration}: {topic} ---")
            log_progress(log_file, f"Source: {url_source}")

            content_stream = []
            
            if topic == "SELF_AWARENESS":
                # Generate local system data instead of web fetch
                system_text = get_local_system_data()
                content_stream = [("SELF", system_text)]
            else:
                # Standard web fetch
                reader = WebCurriculum([url_source])
                content_stream = reader.read_stream()

            for _, content in content_stream:
                log_progress(log_file, f"Content loaded ({len(content)} chars). Studying…")
                if len(content) < 10: continue

                # -------------------------------------------------------
                # Multimodal perception – emotion & thought from first 500 chars
                # -------------------------------------------------------
                short_text = content[:500]
                _, emotion, thought = senses.process_multimodal_input(text=short_text)
                mod_params = senses.get_modulated_parameters()
                current_lr = base_lr * mod_params["k"]
                log_progress(
                    log_file,
                    f"Emotion: {emotion.classify_emotion()} (Valence: {emotion.valence:.2f}, Arousal: {emotion.arousal:.2f})",
                )
                if thought:
                    log_progress(log_file, f"Thought: {thought}")

                # -------------------------------------------------------
                # Pull any new audio tokens and merge with the text
                # -------------------------------------------------------
                audio_tokens = audio_pipe.pop_tokens()
                if audio_tokens:
                    log_progress(log_file, f"Audio tokens captured: {len(audio_tokens)}")
                    content_for_training = content + " " + " ".join(audio_tokens)
                else:
                    content_for_training = content

                # -------------------------------------------------------
                # Train both models on the combined stream
                # -------------------------------------------------------
                res_12d = trainer_12d.train_on_text(
                    content_for_training, gen, seq_len=128, batch_size=16, learning_rate=current_lr
                )
                loss_12d = res_12d["loss"]

                res_42d = trainer_42d.train_on_text(
                    content_for_training, gen, seq_len=128, batch_size=16, learning_rate=current_lr
                )
                loss_42d = res_42d["loss"]

                diff = loss_12d - loss_42d
                winner = "42D" if diff > 0 else "12D"
                log_progress(log_file, f"12D Loss: {loss_12d:.4f}")
                log_progress(log_file, f"42D Loss: {loss_42d:.4f}")
                log_progress(log_file, f"Winner: {winner} (Advantage: {abs(diff):.4f})")

                # -------------------------------------------------------
                # Update brain_state.json for the monitor (include audio count)
                # -------------------------------------------------------
                # Use content_for_training so we see the audio tokens in the stream
                words = content_for_training.split()
                sample_size = min(40, len(words))
                if sample_size > 10:
                    # Prefer showing the end of the sequence where audio tokens are appended
                    start_idx = max(0, len(words) - sample_size)
                    last_tokens = words[start_idx : start_idx + sample_size]
                else:
                    last_tokens = words[:sample_size] if words else ["[No tokens]"]

                brain_state = {
                    "iteration": iteration,
                    "topic": topic,
                    "loss_12d": loss_12d,
                    "loss_42d": loss_42d,
                    "winner": winner,
                    "emotion": emotion.classify_emotion(),
                    "valence": emotion.valence,
                    "arousal": emotion.arousal,
                    "current_thought": thought if thought else "Processing...",
                    "last_tokens": last_tokens,
                    "learning_rate": current_lr,
                    "audio_token_count": len(audio_tokens) if audio_tokens else 0,
                }
                with open(log_dir / "brain_state.json", "w", encoding="utf-8") as f:
                    json.dump(brain_state, f, indent=2)

                # -------------------------------------------------------
                # Save model checkpoints for infinite memory (per iteration)
                # -------------------------------------------------------
                torch.save(model_12d.state_dict(), checkpoint_dir / f"model_12d_iter_{iteration}.pt")
                torch.save(model_42d.state_dict(), checkpoint_dir / f"model_42d_iter_{iteration}.pt")
                log_progress(log_file, f"Checkpoints saved for iteration {iteration}.")

            # Small pause between books – lets CPU breathe & mimics "pondering"
            time.sleep(2)

    except KeyboardInterrupt:
        # Graceful shutdown – stop audio capture first
        audio_pipe.stop()
        log_progress(log_file, "Real‑time audio pipe stopped.")
        log_progress(log_file, "Study session interrupted by user.")
    except Exception as e:
        log_progress(log_file, f"CRITICAL ERROR: {e}")
        raise

if __name__ == "__main__":
    run_autonomous_study()
