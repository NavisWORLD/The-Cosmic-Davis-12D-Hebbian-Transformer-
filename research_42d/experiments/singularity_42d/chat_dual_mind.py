"""
DUAL MIND CHAT INTERFACE
=========================
Talk to both the 12D and 42D minds, watch them debate, and get consensus answers.

Commands:
  /debate  - Trigger a debate between 12D and 42D
  /12d     - Talk only to 12D (grounded)
  /42d     - Talk only to 42D (abstract)
  /both    - Get responses from both
  /save    - Save current state
  /history - Show debate history
  /exit    - Quit
"""

import sys
import torch
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# Add paths
ROOT = Path(__file__).parents[3]
sys.path.append(str(ROOT / "packages" / "cosmic-synapse-transformer"))
sys.path.append(str(Path(__file__).parent))

from cosmic_synapse.data.generate_synthetic_data import SyntheticDataGenerator
from dual_mind_evolution import DualMindDebateSystem, InfiniteMemorySystem

console = Console()

import datetime
import platform

class DualMindChatInterface:
    def __init__(self):
        console.print(Panel.fit(
            "[bold cyan]🌌 DUAL MIND CHAT INTERFACE[/bold cyan]\n"
            "Talk to Past (12D) and Future (42D) simultaneously",
            border_style="cyan"
        ))
        
        # Load or create vocabulary
        self.setup_vocab()
        
        # Initialize dual mind system
        console.print("\n[yellow]Initializing Dual Mind System...[/yellow]")
        self.dual_mind = DualMindDebateSystem(self.gen.vocab_size)
        self.memory = InfiniteMemorySystem()
        
        # Try to load latest checkpoint
        self.load_latest_checkpoint()
        
        self.id_to_word = {v: k for k, v in self.gen.vocab.items()}
        
        console.print("[green]✅ System Ready![/green]\n")
        
    def setup_vocab(self):
        """Load vocabulary"""
        vocab_path = ROOT / "study_session_logs" / "vocab.txt"
        
        self.gen = SyntheticDataGenerator()
        
        if vocab_path.exists():
            console.print(f"[dim]Loading vocabulary from {vocab_path}...[/dim]")
            with open(vocab_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        word, idx = parts
                        self.gen.vocab[word] = int(idx)
            self.gen.vocab_size = len(self.gen.vocab)
        else:
            # Build minimal vocab
            self.gen.build_vocabulary(["hello", "world", "test"], max_vocab_size=1000)
    
    def load_latest_checkpoint(self):
        """Load the most recent checkpoint"""
        history = self.memory.get_checkpoint_history(limit=1)
        if history:
            checkpoint_id, info = history[0]
            console.print(f"[dim]Loading checkpoint: {checkpoint_id}...[/dim]")
            checkpoint = self.memory.load_checkpoint(checkpoint_id)
            if checkpoint:
                self.dual_mind.mind_12d.load_state_dict(checkpoint['models']['12d'])
                self.dual_mind.mind_42d.load_state_dict(checkpoint['models']['42d'])
                console.print(f"[green]✅ Loaded memory from iteration {checkpoint['metadata']['iteration']}[/green]")
    
    def encode(self, text):
        """Encode text to token IDs"""
        ids = []
        for word in text.split(): # Split by whitespace to preserve context formatting
             # Simple normalization
             word_clean = word.lower().strip(".,!?()[]")
             if word_clean in self.gen.vocab:
                ids.append(self.gen.vocab[word_clean])
             else:
                ids.append(0)  # UNK
        return torch.tensor([ids], dtype=torch.long) if ids else torch.tensor([[0]], dtype=torch.long)
    
    def decode(self, ids):
        """Decode token IDs to text"""
        words = []
        for idx in ids[0].tolist():
            if idx in self.id_to_word:
                words.append(self.id_to_word[idx])
        return " ".join(words)
    
    def get_system_context(self):
        """Get current system context for the AI"""
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M")
        os_name = platform.system()
        # "Context: Today is {DATE}. The time is {TIME}. Running on {OS}."
        return f"Context: Today is {date_str} . The time is {time_str} . System is {os_name} . User says: "

    def generate_response(self, model, input_ids, model_name, temperature=0.8, max_tokens=20):
        """Generate and display response from a model"""
        model.eval()
        
        with torch.no_grad():
            # Get initial logits
            logits, _, _ = model(input_ids)
            
            # Sample from logits
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Start with input and add token
            output = torch.cat([input_ids, next_token], dim=1)
            
            # Generate more tokens
            for _ in range(max_tokens - 1):
                logits, _, _ = model(output)
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                output = torch.cat([output, next_token], dim=1)
        
        # Only decode the *new* tokens (response), not the full context
        full_decoded = self.decode(output)
        
        # Heuristic: split by "says:" to find response if context was used
        if "says:" in full_decoded:
            response = full_decoded.split("says:")[-1].strip()
        else:
            response = full_decoded # Fallback
        
        # Display with color
        color = "cyan" if model_name == "12D" else "magenta"
        console.print(f"\n[{color}]🧠 {model_name} MIND:[/{color}]", end=" ")
        
        for word in response.split():
            console.print(f"[{color}]{word}[/{color}]", end=" ")
            time.sleep(0.05)
        console.print()
        
        return response
    
    def trigger_debate(self, prompt):
        """Trigger a debate between the two minds"""
        console.print("\n[bold yellow]⚔️  INITIATING DEBATE MODE...[/bold yellow]\n")
        
        context = self.get_system_context()
        full_prompt = context + prompt
        input_ids = self.encode(full_prompt)
        
        try:
            debate = self.dual_mind.debate(input_ids, rounds=2)
            
            if 'error' in debate:
                console.print(f"[red]Debate error: {debate['error']}[/red]")
                return
            
            # Display debate
            for round_num, round_data in enumerate(debate['rounds']):
                console.print(f"\n[bold]Round {round_num + 1}:[/bold]")
                
                # 12D response - already a list from tolist()
                response_12d_list = round_data['12d_response']
                # Convert back to tensor properly
                if isinstance(response_12d_list[0], list):
                    # It's a nested list [[tokens]]
                    response_12d = torch.tensor(response_12d_list)
                else:
                    # It's a flat list [tokens]
                    response_12d = torch.tensor([response_12d_list])
                    
                decoded_12d = self.decode(response_12d)
                console.print(f"[cyan]  12D (Grounded): {decoded_12d}[/cyan]")
                
                # 42D response
                response_42d_list = round_data['42d_response']
                if isinstance(response_42d_list[0], list):
                    response_42d = torch.tensor(response_42d_list)
                else:
                    response_42d = torch.tensor([response_42d_list])
                    
                decoded_42d = self.decode(response_42d)
                console.print(f"[magenta]  42D (Abstract): {decoded_42d}[/magenta]")
            
            # Consensus
            consensus_list = debate['consensus']
            if isinstance(consensus_list[0], list):
                consensus = torch.tensor(consensus_list)
            else:
                consensus = torch.tensor([consensus_list])
                
            decoded_consensus = self.decode(consensus)
            console.print(f"\n[bold green]✨ CONSENSUS:[/bold green] {decoded_consensus}")
            
        except Exception as e:
            console.print(f"[red]Error during debate: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    def show_commands(self):
        """Show available commands"""
        table = Table(title="Available Commands", border_style="dim")
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        
        table.add_row("/debate", "Watch 12D and 42D debate")
        table.add_row("/12d", "Talk only to 12D (grounded)")
        table.add_row("/42d", "Talk only to 42D (abstract)")
        table.add_row("/both", "Get responses from both")
        table.add_row("/save", "Save current brain state")
        table.add_row("/history", "Show recent debates")
        table.add_row("/help", "Show this help")
        table.add_row("/exit", "Quit")
        
        console.print(table)
    
    def run(self):
        """Main chat loop"""
        self.show_commands()
        
        while True:
            try:
                user_input = console.input("\n[bold green]👤 YOU:[/bold green] ")
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    cmd = user_input.split()[0].lower()
                    
                    if cmd in ["/exit", "/quit"]:
                        console.print("[yellow]Goodbye! 🌌[/yellow]")
                        break
                    
                    elif cmd == "/help":
                        self.show_commands()
                        continue
                    
                    elif cmd == "/debate":
                        prompt = " ".join(user_input.split()[1:]) if len(user_input.split()) > 1 else "what is consciousness"
                        self.trigger_debate(prompt)
                        continue
                    
                    elif cmd == "/save":
                        checkpoint_id = self.memory.create_checkpoint(
                            models_state={
                                '12d': self.dual_mind.mind_12d.state_dict(),
                                '42d': self.dual_mind.mind_42d.state_dict()
                            },
                            metadata={'manual_save': True}
                        )
                        console.print(f"[green]💾 State saved: {checkpoint_id}[/green]")
                        continue
                    
                    elif cmd == "/history":
                        history = self.memory.get_checkpoint_history()
                        console.print(f"\n[bold]Recent Checkpoints:[/bold]")
                        for cp_id, info in history[:5]:
                            console.print(f"  - {cp_id} ({info['timestamp']})")
                        continue
                    
                    elif cmd == "/12d":
                        prompt = " ".join(user_input.split()[1:])
                        context = self.get_system_context()
                        input_ids = self.encode(context + prompt)
                        self.generate_response(self.dual_mind.mind_12d, input_ids, "12D", temperature=0.7)
                        continue
                    
                    elif cmd == "/42d":
                        prompt = " ".join(user_input.split()[1:])
                        context = self.get_system_context()
                        input_ids = self.encode(context + prompt)
                        self.generate_response(self.dual_mind.mind_42d, input_ids, "42D", temperature=0.9)
                        continue
                    
                    elif cmd == "/both":
                        prompt = " ".join(user_input.split()[1:])
                        context = self.get_system_context()
                        input_ids = self.encode(context + prompt)
                        self.generate_response(self.dual_mind.mind_12d, input_ids, "12D", temperature=0.7)
                        self.generate_response(self.dual_mind.mind_42d, input_ids, "42D", temperature=0.9)
                        continue
                
                # Default: get both responses
                context = self.get_system_context()
                input_ids = self.encode(context + user_input)
                self.generate_response(self.dual_mind.mind_12d, input_ids, "12D", temperature=0.7)
                self.generate_response(self.dual_mind.mind_42d, input_ids, "42D", temperature=0.9)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye! 🌌[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    interface = DualMindChatInterface()
    interface.run()
