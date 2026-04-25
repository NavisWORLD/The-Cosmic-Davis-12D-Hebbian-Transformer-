"""
NEURAL MONITOR – PROFESSIONAL REAL‑TIME DASHBOARD
================================================
A terminal UI that visualises the internal state of the Cosmic Davis multimodal AI.
Features:
*   Live loss curves for 12D and 42D models (ASCII spark‑line).
*   Scrolling token stream (EEG‑style) with deterministic colour hashing.
*   12‑dimensional state matrix visualised as animated bars.
*   Emotion panel (valence / arousal) with dynamic colour.
*   Thought panel showing the latest internal monologue.
*   Robust file‑watching – only reloads when `brain_state.json`
    changes, avoiding unnecessary I/O.
*   Configurable refresh rate, history length and buffer sizes.
"""

import json
import time
import math
import random
from pathlib import Path
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.table import Table

# ---------------------------------------------------------------------------
# Configuration (tweak to your taste)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parents[1]
LOG_DIR = ROOT_DIR / "study_session_logs"
STATE_FILE = LOG_DIR / "brain_state.json"
MAX_HISTORY = 80               # Number of loss points to keep for the spark‑line
TOKEN_BUFFER_SIZE = 300        # Max tokens stored for scrolling view
TOKEN_DISPLAY = 120            # Tokens shown in the panel at once
REFRESH_RATE = 0.2             # Seconds between UI refreshes (higher = smoother)

console = Console()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def colour_hash(token: str) -> str:
    """Deterministic colour for a token – works well on dark terminals."""
    palette = [
        "bright_green",
        "green",
        "spring_green1",
        "cyan",
        "bright_cyan",
        "magenta",
        "bright_magenta",
        "yellow",
    ]
    return palette[hash(token) % len(palette)]

# ---------------------------------------------------------------------------
# NeuralMonitor – core UI logic
# ---------------------------------------------------------------------------
class NeuralMonitor:
    def __init__(self):
        # Loss history for the spark‑line
        self.loss_12d = deque(maxlen=MAX_HISTORY)
        self.loss_42d = deque(maxlen=MAX_HISTORY)
        # Token rate history
        self.token_rates = deque(maxlen=MAX_HISTORY)
        self.last_token_count = 0
        self.last_update_time = time.time()
        
        # Token scrolling buffer
        self.token_buffer = deque(maxlen=TOKEN_BUFFER_SIZE)
        self.last_seen_tokens = []
        # File‑watching guard
        self.last_mtime = 0.0
        # Wave phase for the 12D matrix animation
        self.wave_phase = 0.0

    # ---------------------------------------------------------------------
    # Load brain_state.json only when it has changed
    # ---------------------------------------------------------------------
    def load_state(self):
        try:
            if not STATE_FILE.exists():
                return None
            mtime = STATE_FILE.stat().st_mtime
            if mtime <= self.last_mtime:
                return None  # No new data
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.last_mtime = mtime
            return state
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Update internal buffers from the latest state
    # ---------------------------------------------------------------------
    def update_buffers(self, state):
        # Loss history
        self.loss_12d.append(state.get("loss_12d", 0.0))
        self.loss_42d.append(state.get("loss_42d", 0.0))
        
        # Calculate Token Rate
        current_count = state.get("audio_token_count", 0)
        current_time = time.time()
        time_diff = current_time - self.last_update_time
        
        if time_diff > 0 and self.last_token_count > 0:
            # Tokens per second
            rate = (current_count - self.last_token_count) / time_diff
            # Smooth it a bit
            rate = max(0, min(rate, 1000)) # Cap at 1000 to prevent spikes
            self.token_rates.append(rate)
        else:
            self.token_rates.append(0)
            
        self.last_token_count = current_count
        self.last_update_time = current_time

        # Token stream – only add when the list changes
        new_tokens = state.get("last_tokens", [])
        if new_tokens and new_tokens != self.last_seen_tokens:
            self.token_buffer.extend(new_tokens)
            self.last_seen_tokens = new_tokens

    # ---------------------------------------------------------------------
    # UI components
    # ---------------------------------------------------------------------
    def loss_graph(self) -> Text:
        """Render an 8‑row ASCII spark‑line for the two loss curves."""
        height = 8
        if not self.loss_12d:
            return Text("Waiting for loss data…", style="dim")
        all_vals = list(self.loss_12d) + list(self.loss_42d)
        if not all_vals: return Text("No data", style="dim")
        
        min_v, max_v = min(all_vals), max(all_vals)
        range_v = max_v - min_v if max_v != min_v else 1.0
        lines = []
        for row in range(height):
            threshold = max_v - (range_v * row / height)
            line = ""
            for l12, l42 in zip(self.loss_12d, self.loss_42d):
                if l12 >= threshold and l42 >= threshold:
                    char = "█"  # both models strong
                elif l12 >= threshold:
                    char = "░"  # 12D only
                elif l42 >= threshold:
                    char = "▓"  # 42D only
                else:
                    char = " "
                line += char
            lines.append(line)
        return Text("\n".join(lines), style="green")

    def token_rate_graph(self) -> Text:
        """Render an 8-row ASCII spark-line for token rate."""
        height = 8
        if not self.token_rates:
            return Text("Calculating rate...", style="dim")
            
        vals = list(self.token_rates)
        min_v, max_v = min(vals), max(vals)
        range_v = max_v - min_v if max_v != min_v else 1.0
        
        lines = []
        for row in range(height):
            threshold = max_v - (range_v * row / height)
            line = ""
            for val in vals:
                if val >= threshold:
                    char = "█"
                elif val >= threshold - (range_v/height/2):
                    char = "▄"
                else:
                    char = " "
                line += char
            lines.append(line)
        
        current_rate = vals[-1] if vals else 0
        return Text(f"Current: {current_rate:.1f} tok/s\n" + "\n".join(lines), style="cyan")

    def token_panel(self, state) -> Panel:
        audio_count = state.get("audio_token_count", 0)
        
        if not self.token_buffer:
            txt = Text("No tokens yet – start training to see the stream…", style="dim yellow")
        else:
            txt = Text()
            # Show last N tokens
            display = list(self.token_buffer)[-TOKEN_DISPLAY:]
            for tok in display:
                if tok.startswith("<freq_"):
                    # Parse frequency index
                    try:
                        freq_idx = int(tok.split('_')[1].strip('>'))
                        # Map frequency to color (Rainbow spectrum)
                        # 0-10: Red/Orange (Low)
                        # 10-20: Yellow/Green (Mid)
                        # 20-32: Blue/Violet (High)
                        if freq_idx < 10:
                            style = "bold red"
                            symbol = "🔴"
                        elif freq_idx < 20:
                            style = "bold yellow"
                            symbol = "🟡"
                        else:
                            style = "bold blue"
                            symbol = "🔵"
                        
                        tok_display = f"{symbol}{tok}"
                    except:
                        style = "bold magenta"
                        tok_display = f"♫{tok}"
                else:
                    # Text token style - Deterministic color
                    style = colour_hash(tok)
                    tok_display = tok
                txt.append(f"{tok_display} ", style=style)
        
        title = f"🌊 Token Stream | Audio Tokens: {audio_count}"
        return Panel(txt, title=title, border_style="bright_green")

    def audio_metrics_panel(self, state) -> Panel:
        """Show live audio conversion metrics."""
        audio_tokens = [t for t in state.get("last_tokens", []) if t.startswith("<freq_")]
        
        if not audio_tokens:
            return Panel(Text("No audio input detected...", style="dim"), title="🎵 Audio Cortex", border_style="dim")
            
        # Calculate metrics from recent tokens
        freq_indices = []
        for t in audio_tokens:
            try:
                freq_indices.append(int(t.split('_')[1].strip('>')))
            except:
                pass
                
        if not freq_indices:
             return Panel(Text("Waiting for frequency data...", style="dim"), title="🎵 Audio Cortex", border_style="dim")

        avg_freq = sum(freq_indices) / len(freq_indices)
        max_freq = max(freq_indices)
        energy = len(freq_indices) / 10.0 # Rough energy metric
        
        # 12D Metric Mapping
        x12_metric = (avg_freq / 32.0) * 2 - 1 # Map 0-32 to -1 to 1
        
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="cyan")
        table.add_column(justify="left", style="white")
        
        table.add_row("Avg Freq:", f"{avg_freq:.1f} Hz (Index)")
        table.add_row("Peak Freq:", f"{max_freq} (Index)")
        table.add_row("Pattern:", "🌊 Harmonic" if len(set(freq_indices)) < 5 else "⚡ Chaotic")
        table.add_row("Energy:", "█" * int(energy))
        table.add_row("X12 Metric:", f"{x12_metric:.4f} (Valence)")
        
        return Panel(table, title="🎵 Audio Conversion Metrics", border_style="magenta")

    def thought_panel(self, state) -> Panel:
        thought = state.get("current_thought", "[no thought]")
        txt = Text(thought, style="italic cyan")
        return Panel(txt, title="💭 Internal Monologue", border_style="cyan")

    def emotion_panel(self, state) -> Panel:
        emotion = state.get("emotion", "neutral")
        valence = state.get("valence", 0.0)
        arousal = state.get("arousal", 0.0)
        if valence > 0.3:
            col = "green"
        elif valence < -0.3:
            col = "red"
        else:
            col = "white"
        body = Align.center(
            f"{emotion.upper()}\n\nValence: {valence:.2f}\nArousal: {arousal:.2f}",
            vertical="middle",
        )
        return Panel(body, title="❤️ Emotion Core", border_style=col)

    def matrix_panel(self, state) -> Panel:
        dims = [
            "D1 Energy",
            "D2 Mass",
            "D3 Phi",
            "D4 Chaos",
            "D5 Vel‑X",
            "D6 Vel‑Y",
            "D7 Vel‑Z",
            "D8 Connect",
            "D9 Cosmic",
            "D10 Entropy",
            "D11 Freq",
            "D12 Self",
        ]
        valence = state.get("valence", 0.0)
        arousal = state.get("arousal", 0.5)
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right")
        table.add_column()
        for i, dim in enumerate(dims):
            wave = (math.sin(self.wave_phase + i) + 1) / 2
            value = wave * (0.5 + arousal * 0.5) + random.random() * 0.1
            value = max(0.0, min(1.0, value))
            bar_len = int(value * 20)
            bar = "━" * bar_len
            if i == 3:
                col = "red"   # Chaos
            elif i == 11:
                col = "cyan"  # Self
            elif i == 2:
                col = "gold1" # Phi
            else:
                col = "green"
            table.add_row(dim, Text(bar, style=col))
        self.wave_phase += 0.2
        return Panel(table, title="🌌 12D State Matrix", border_style="magenta")

    def header(self, state) -> Panel:
        iter_num = state.get("iteration", 0)
        topic = state.get("topic", "unknown")
        txt = Align.center(
            f"🧠 COSMIC DAVIS | Iteration: {iter_num} | Topic: {topic}",
            vertical="middle",
        )
        return Panel(txt, style="bold white on blue")

    def footer(self) -> Panel:
        return Panel(
            Align.center("Press Ctrl+C to exit – training continues in background", vertical="middle"),
            style="dim",
        )

    # ---------------------------------------------------------------------
    # Layout assembly
    # ---------------------------------------------------------------------
    def make_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )
        # Body: left (graph + tokens) | right (matrix + emotion + thought)
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )
        layout["left"].split(
            Layout(name="graph", ratio=2),
            Layout(name="audio_metrics", ratio=1),
            Layout(name="tokens", ratio=2),
        )
        layout["right"].split(
            Layout(name="matrix", ratio=2),
            Layout(name="emotion", ratio=1),
            Layout(name="thought", ratio=1),
        )
        return layout

    # ---------------------------------------------------------------------
    # Main loop – refreshes at REFRESH_RATE
    # ---------------------------------------------------------------------
    def run(self):
        layout = self.make_layout()
        with Live(layout, refresh_per_second=int(1 / REFRESH_RATE), screen=True):
            while True:
                state = self.load_state()
                if state:
                    self.update_buffers(state)
                # Header (needs state – fall back to empty dict)
                layout["header"].update(self.header(state or {}))
                # Graph
                layout["graph"].update(
                    Panel(self.loss_graph(), title="📉 Loss Curve", border_style="green")
                )
                # Audio Metrics
                layout["audio_metrics"].update(self.audio_metrics_panel(state or {}))
                # Tokens
                layout["tokens"].update(self.token_panel(state or {}))
                # Right side panels (use latest state or empty dict)
                layout["matrix"].update(self.matrix_panel(state or {}))
                layout["emotion"].update(self.emotion_panel(state or {}))
                layout["thought"].update(self.thought_panel(state or {}))
                # Footer
                layout["footer"].update(self.footer())
                time.sleep(REFRESH_RATE)

if __name__ == "__main__":
    try:
        NeuralMonitor().run()
    except KeyboardInterrupt:
        console.print("\n[bold red]Neural monitor stopped.[/]")
