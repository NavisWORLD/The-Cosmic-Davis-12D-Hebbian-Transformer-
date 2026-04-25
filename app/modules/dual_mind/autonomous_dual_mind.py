"""
AUTONOMOUS DUAL MIND EVOLUTION
================================
Fully autonomous training where 12D and 42D learn together,
debate periodically, and evolve their understanding.

Features:
- Continuous autonomous learning
- Periodic debates (every 10 iterations)
- Auto-checkpointing (every 100 iterations)
- Infinite memory system
- Brain state monitoring
"""

import sys
import torch
import random
import json
import subprocess
import platform
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parents[3] / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(Path(__file__).parents[3] / "app" / "modules" / "dual_mind"))
sys.path.append(str(Path(__file__).parents[3] / "app" / "modules" / "audio"))

from cosmic_synapse.data.web_reader import WebCurriculum
from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from cosmic_synapse.training.unified_trainer import UnifiedCosmicTrainer
from dual_mind_evolution import EvolutionaryLearningLoop, DualMindDebateSystem, InfiniteMemorySystem
from multimodal_fusion import UnifiedMultimodalSystem

# Curriculum
TOPICS = {
    "LITERATURE": [
        "https://www.gutenberg.org/files/11/11-0.txt",
    ],
    "PHILOSOPHY": [
        "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
    ],
    "SCIENCE": [
        "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
    ],
    "LIVE_WORLD": [
        "https://text.npr.org/", # Lightweight text news
        "https://lite.cnn.com", # Text-heavy news
    ],
    "SELF_AWARENESS": [
        "LOCAL_SYSTEM_SCAN", # Special token for generating local data
    ],
    # NEW: Conversational learning - teaches the model how to chat
    "CONVERSATION": [
        "LOCAL_CONVERSATION_SEED", # Bootstrapped Q&A pairs
        "LOCAL_CONVERSATION_LOG",  # Live log from user chats
    ]
}

def log_progress(log_file, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with open(log_file, "a", encoding='utf-8') as f:
        f.write(entry + "\\n")

def get_local_system_data():
    """Generate a text report of the local system status for grounding."""
    report = []
    report.append(f"SYSTEM_TIMESTAMP: {datetime.now().isoformat()}")
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

def run_dual_mind_evolution():
    """Main autonomous evolution loop"""
    
    log_dir = Path("dual_mind_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "evolution_log.txt"
    
    log_progress(log_file, "="*60)
    log_progress(log_file, "🧠 DUAL MIND EVOLUTIONARY SYSTEM STARTING")
    log_progress(log_file, "12D (Past) ⇄ 42D (Future) → Present")
    log_progress(log_file, "="*60)
    
    # Build vocabulary
    log_progress(log_file, "Building unified vocabulary...")
    gen = SyntheticDataGenerator()
    
    vocab_path = Path("study_session_logs") / "vocab.txt"
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word, idx = parts
                    gen.vocab[word] = int(idx)
        gen.vocab_size = len(gen.vocab)
    else:
        # Quick vocab build
        gen.build_vocabulary(["test", "hello", "world"], max_vocab_size=10000)
    
    log_progress(log_file, f"Vocabulary: {gen.vocab_size} tokens")
    
    # Initialize Evolution System
    evolution_loop = EvolutionaryLearningLoop(gen.vocab_size)
    
    # Initialize trainers
    trainer_12d = UnifiedCosmicTrainer(
        evolution_loop.dual_mind.mind_12d,
        evolution_loop.dual_mind.mind_12d.config,
        model_type="12D"
    )
    
    trainer_42d = UnifiedCosmicTrainer(
        evolution_loop.dual_mind.mind_42d,
        evolution_loop.dual_mind.mind_42d.config,
        model_type="42D"
    )
    
    # Multimodal senses
    senses = UnifiedMultimodalSystem()
    base_lr = 3e-4 * 0.618
    
    log_progress(log_file, "Systems initialized. Beginning evolution...")
    
    iteration = 0
    debate_count = 0
    
    try:
        while True:
            iteration += 1
            
            # Pick random topic
            topic = random.choice(list(TOPICS.keys()))
            url_source = random.choice(TOPICS[topic])
            
            log_progress(log_file, f"\\n--- ITERATION {iteration}: {topic} ---")
            
            content_stream = []
            
            if topic == "SELF_AWARENESS":
                 system_text = get_local_system_data()
                 # Wrap in list of tuples for compatibility with loop below
                 content_stream = [("SELF", system_text)]
            elif topic == "CONVERSATION":
                 # Load conversational training data from local files
                 conv_seed_path = Path("study_session_logs") / "conversation_seed.txt"
                 conv_log_path = Path("study_session_logs") / "conversation_log.txt"
                 
                 conv_text = ""
                 if url_source == "LOCAL_CONVERSATION_SEED" and conv_seed_path.exists():
                     conv_text = conv_seed_path.read_text(encoding="utf-8")
                     log_progress(log_file, f"Loaded conversation seed: {len(conv_text)} chars")
                 elif url_source == "LOCAL_CONVERSATION_LOG" and conv_log_path.exists():
                     conv_text = conv_log_path.read_text(encoding="utf-8")
                     log_progress(log_file, f"Loaded live conversation log: {len(conv_text)} chars")
                 else:
                     # Fallback: generate a simple conversation prompt
                     conv_text = "Q: Hello. A: Hello! How can I help you today?"
                 
                 content_stream = [("CONVERSATION", conv_text)]
            else:
                 reader = WebCurriculum([url_source])
                 content_stream = reader.read_stream()
            
            for _, content in content_stream:
                log_progress(log_file, f"Content: {len(content)} chars")
                if len(content) < 10: continue

                # Emotion detection
                short_text = content[:500]
                _, emotion, thought = senses.process_multimodal_input(text=short_text)
                mod_params = senses.get_modulated_parameters()
                current_lr = base_lr * mod_params['k']
                
                log_progress(log_file, f"Emotion: {emotion.classify_emotion()}")
                
                # Train both minds
                res_12d = trainer_12d.train_on_text(content, gen, seq_len=128, batch_size=8, learning_rate=current_lr)
                res_42d = trainer_42d.train_on_text(content, gen, seq_len=128, batch_size=8, learning_rate=current_lr)
                
                loss_12d = res_12d['loss']
                loss_42d = res_42d['loss']
                
                log_progress(log_file, f"12D Loss: {loss_12d:.4f} | 42D Loss: {loss_42d:.4f}")
                
                # Debate phase (every 10 iterations)
                if iteration % 10 == 0:
                    log_progress(log_file, "⚔️  DEBATE MODE ACTIVATED")
                    
                    # Create debate prompt from content
                    words = content.split()[:20]
                    debate_ids = torch.tensor([[gen.vocab.get(w.lower(), 0) for w in words]])
                    
                    debate = evolution_loop.dual_mind.debate(debate_ids, rounds=2)
                    debate_count += 1
                    
                    log_progress(log_file, f"Debate #{debate_count} completed - {len(debate['rounds'])} rounds")
                
                # Auto-checkpoint (every 100 iterations)
                if iteration % 100 == 0:
                    checkpoint_id = evolution_loop.save_state()
                    log_progress(log_file, f"💾 CHECKPOINT: {checkpoint_id}")
                
                # Update brain state for monitor
                brain_state = {
                    "iteration": iteration,
                    "topic": topic,
                    "loss_12d": loss_12d,
                    "loss_42d": loss_42d,
                    "emotion": emotion.classify_emotion(),
                    "valence": emotion.valence,
                    "arousal": emotion.arousal,
                    "current_thought": thought if thought else "Evolving...",
                    "debates_total": debate_count,
                    "last_checkpoint": iteration - (iteration % 100),
                    "learning_rate": current_lr
                }
                
                with open(log_dir / "brain_state.json", "w") as f:
                    json.dump(brain_state, f, indent=2)
                
                break  # One content per iteration
    
    except KeyboardInterrupt:
        log_progress(log_file, "\\n🛑 Evolution interrupted by user")
        # Final save
        final_checkpoint = evolution_loop.save_state()
        log_progress(log_file, f"💾 Final checkpoint: {final_checkpoint}")
    except Exception as e:
        log_progress(log_file, f"\\n❌ ERROR: {e}")
        raise e



if __name__ == "__main__":
    run_dual_mind_evolution()
