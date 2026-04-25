"""
COSMO'S cosmos PLUGIN
===========================

Proper cosmos Plugin subclass that exposes the Cosmo's Transformer
as tools, agents, and processors within the cosmos ecosystem.

Lifecycle:
    initialize() → Load model, restore checkpoint
    activate()   → Register tools + agents
    deactivate() → Save checkpoint, free GPU
    shutdown()   → Clean up everything

Capabilities:
    - Agent:     CosmosAgent (uses transformer for generation)
    - Tool:      cosmos_generate, cosmos_train_text, cosmos_state
    - Processor: 54D emotional state + CST bridge integration

Author: Cory Shane Davis
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure cosmos is importable
try:
    from cosmos.plugins.base import Plugin, PluginMetadata, PluginCapability, PluginStatus
    cosmos_AVAILABLE = True
except ImportError:
    # Standalone mode — create stubs
    cosmos_AVAILABLE = False

    class PluginCapability:
        AGENT = "agent"
        TOOL = "tool"
        PROCESSOR = "processor"

    class PluginStatus:
        UNLOADED = "unloaded"
        LOADED = "loaded"
        ACTIVE = "active"
        ERROR = "error"

    class PluginMetadata:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Plugin:
        """Stub Plugin base for standalone mode."""
        metadata: PluginMetadata = PluginMetadata(
            name="stub", version="0.0.0", capabilities=[]
        )

        def __init__(self, config=None):
            self._status = PluginStatus.UNLOADED
            self._config = config or {}
            self._tools = {}
            self._agents = {}
            self._hooks = {}

        @property
        def id(self): return "stub"
        @property
        def name(self): return self.metadata.name
        @property
        def version(self): return self.metadata.version

        def initialize(self): return True
        def activate(self): return True
        def deactivate(self): return True
        def shutdown(self): return True
        def get_tools(self): return self._tools
        def get_agents(self): return self._agents
        def get_hooks(self): return self._hooks
        def register_tool(self, name, handler): self._tools[name] = handler
        def register_agent(self, name, agent_class): self._agents[name] = agent_class
        def register_hook(self, hook, handler): self._hooks.setdefault(hook, []).append(handler)
        def get_config(self, key, default=None): return self._config.get(key, default)
        def set_config(self, key, value): self._config[key] = value
        def validate_config(self): return True


# Import Cosmo's model components
from .model.cosmos_config import CosmosConfig
from .model.cosmos_model import CosmosTransformer

# Try to import engine components
try:
    from .engine.cst_sensory_bridge import CSTSensoryBridge
    SENSORY_AVAILABLE = True
except ImportError:
    SENSORY_AVAILABLE = False

try:
    from .consciousness.internal_monologue import InternalMonologue
    MONOLOGUE_AVAILABLE = True
except ImportError:
    MONOLOGUE_AVAILABLE = False


class CosmosPlugin(Plugin):
    """
    cosmos Plugin exposing Cosmo's Transformer.

    Provides:
    - cosmos_generate: Generate text using the 54D transformer
    - cosmos_train_text: Train on text with CST dynamics
    - cosmos_state: Inspect the 54D internal state
    - CosmosAgent: Full agent powered by the model
    """

    metadata = PluginMetadata(
        name="Cosmo's",
        version="2.0.0",
        author="Cory Shane Davis",
        description=(
            "Cosmo's — The Unified 54D CST Transformer. "
            "Fuses 12D Cosmic Synapse Theory and 42D Hyper-Cosmic Theory "
            "into a single, superior architecture that surpasses cosmos "
            "at every level."
        ),
        capabilities=[
            PluginCapability.AGENT,
            PluginCapability.TOOL,
            PluginCapability.PROCESSOR,
        ],
        dependencies=[],
        homepage="https://github.com/NavisWORLD/The-Cosmic-Davis-12D-Hebbian-Transformer",
        license="MIT",
    )

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config)
        self.model: Optional[CosmosTransformer] = None
        self.persistent_state = None
        self.tokenizer = None
        self._device = None

        # Plugin-specific config defaults
        default_config = {
            "checkpoint_path": "checkpoints/cosmos/cosmos_best.pt",
            "model_preset": "tiny",  # tiny, small, medium, large
            "device": "auto",
            "temperature": 0.8,
            "top_k": 50,
            "max_gen_tokens": 200,
        }
        if config:
            default_config.update(config)
        self._config = default_config

    def initialize(self) -> bool:
        """
        Load and initialize the Cosmo's model.

        Attempts to load from checkpoint; falls back to fresh model.
        """
        try:
            import torch
            print("[COSMO'S PLUGIN] Initializing...")

            # Determine device
            device_str = self.get_config("device", "auto")
            if device_str == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(device_str)

            # Attempt checkpoint load
            ckpt_path = Path(self.get_config("checkpoint_path", ""))
            if ckpt_path.exists():
                print(f"[COSMO'S PLUGIN] Loading checkpoint: {ckpt_path}")
                checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                cfg_dict = checkpoint["config"]
                config = CosmosConfig(
                    vocab_size=cfg_dict["vocab_size"],
                    d_model=cfg_dict["d_model"],
                    n_layers=cfg_dict["n_layers"],
                    n_heads=cfg_dict["n_heads"],
                )
                self.model = CosmosTransformer(config)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.persistent_state = checkpoint.get("persistent_state")
            else:
                # Fresh model from preset
                preset = self.get_config("model_preset", "tiny")
                preset_fn = getattr(CosmosConfig, preset, CosmosConfig.tiny)
                config = preset_fn()
                self.model = CosmosTransformer(config)
                print(f"[COSMO'S PLUGIN] Created fresh model ({preset} preset)")

            self.model = self.model.to(self._device)
            self.model.eval()

            # Setup tokenizer
            self._setup_tokenizer(config.vocab_size)

            self._status = PluginStatus.LOADED
            print(f"[COSMO'S PLUGIN] Model loaded on {self._device}")
            return True

        except Exception as e:
            print(f"[COSMO'S PLUGIN] Initialization failed: {e}")
            self._status = PluginStatus.ERROR
            return False

    def activate(self) -> bool:
        """Register tools and agents with cosmos."""
        try:
            # Register tools
            self.register_tool("cosmos_generate", self._tool_generate)
            self.register_tool("cosmos_train_text", self._tool_train_text)
            self.register_tool("cosmos_state", self._tool_state)

            # Register hooks
            self.register_hook("pre_response", self._hook_pre_response)
            self.register_hook("post_response", self._hook_post_response)

            self._status = PluginStatus.ACTIVE
            print("[COSMO'S PLUGIN] Activated — tools registered")
            return True

        except Exception as e:
            print(f"[COSMO'S PLUGIN] Activation failed: {e}")
            self._status = PluginStatus.ERROR
            return False

    def deactivate(self) -> bool:
        """Save checkpoint and release resources."""
        try:
            if self.model is not None:
                # Save checkpoint
                import torch
                ckpt_dir = Path("checkpoints/cosmos")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / "cosmos_plugin_state.pt"
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "config": {
                        "vocab_size": self.model.config.vocab_size,
                        "d_model": self.model.config.d_model,
                        "n_layers": self.model.config.n_layers,
                        "n_heads": self.model.config.n_heads,
                    },
                    "persistent_state": self.persistent_state,
                }, ckpt_path)
                print(f"[COSMO'S PLUGIN] State saved to {ckpt_path}")

            self._status = PluginStatus.LOADED
            return True

        except Exception as e:
            print(f"[COSMO'S PLUGIN] Deactivation error: {e}")
            return False

    def shutdown(self) -> bool:
        """Full cleanup."""
        self.deactivate()
        self.model = None
        self.persistent_state = None
        self._status = PluginStatus.UNLOADED
        print("[COSMO'S PLUGIN] Shutdown complete")
        return True

    # ---- TOOLS ----

    def _tool_generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using Cosmo's Transformer."""
        import torch

        if self.model is None:
            return {"error": "Model not initialized"}

        temperature = kwargs.get("temperature", self.get_config("temperature", 0.8))
        top_k = kwargs.get("top_k", self.get_config("top_k", 50))
        max_tokens = kwargs.get("max_tokens", self.get_config("max_gen_tokens", 200))

        # Tokenize
        tokens = self._tokenize(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=self._device)

        # Generate
        with torch.no_grad():
            output, new_state = self.model.generate(
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                state_x54=self.persistent_state,
            )

        # Update persistent state
        self.persistent_state = new_state

        # Decode
        gen_tokens = output[0, len(tokens):].tolist()
        text = self._decode(gen_tokens)

        return {
            "text": text,
            "tokens_generated": len(gen_tokens),
            "x54_mean": new_state.mean().item(),
            "x54_std": new_state.std().item(),
        }

    def _tool_train_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Train the model on text."""
        if self.model is None:
            return {"error": "Model not initialized"}

        from .model.cosmos_trainer import CosmosTrainer

        self.model.train()
        trainer = CosmosTrainer(
            model=self.model,
            lr=kwargs.get("lr", 3e-4),
            device=str(self._device),
        )
        trainer.persistent_state = self.persistent_state

        result = trainer.train_on_text(
            text,
            batch_size=kwargs.get("batch_size", 2),
            seq_len=kwargs.get("seq_len", 128),
            n_epochs=kwargs.get("n_epochs", 1),
        )

        self.persistent_state = trainer.persistent_state
        self.model.eval()

        return result

    def _tool_state(self) -> Dict[str, Any]:
        """Inspect the current 54D internal state."""
        if self.persistent_state is None:
            return {"status": "No persistent state — model hasn't been run yet"}

        s = self.persistent_state
        return {
            "shape": list(s.shape),
            "mean": s.mean().item(),
            "std": s.std().item(),
            "scalar_mean": s[:, :, :12].mean().item(),
            "vector_mean": s[:, :, 12:].mean().item(),
            "min": s.min().item(),
            "max": s.max().item(),
        }

    # ---- HOOKS ----

    def _hook_pre_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-response hook: inject 54D state metrics into context."""
        if self.persistent_state is not None:
            context["cosmos_state"] = self._tool_state()
        return context

    def _hook_post_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Post-response hook: log the interaction for dream processing."""
        # Future: feed to DreamProcessor for consolidation
        return context

    # ---- TOKENIZER ----

    def _setup_tokenizer(self, vocab_size: int) -> None:
        """Initialize tokenizer."""
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self._tokenize = self.tokenizer.encode
            self._decode = self.tokenizer.decode
        except ImportError:
            chars = list(set(
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                " .,!?;:'\"-—()[]{}/@#$%^&*+=<>~`\n\t"
            ))
            stoi = {ch: i % vocab_size for i, ch in enumerate(chars)}
            itos = {i: ch for ch, i in stoi.items()}
            self._tokenize = lambda text: [stoi.get(c, 0) for c in text]
            self._decode = lambda tokens: "".join(itos.get(t, "?") for t in tokens)


# ===================================================================
# CONVENIENCE: Direct Plugin Registration
# ===================================================================

def get_plugin() -> CosmosPlugin:
    """
    Entry point for cosmos plugin discovery.

    cosmos's PluginLoader can call this function to get the plugin.
    """
    return CosmosPlugin()
