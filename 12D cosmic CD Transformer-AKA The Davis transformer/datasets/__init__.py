"""
Datasets Module for 12D Cosmic Synapse Transformer

This module provides synthetic data generation capabilities for training
the Cosmic Synapse Transformer with $0 cost.

Available generators:
- MarkovGenerator: Text generation using Markov chains
- GrammarGenerator: Context-free grammar-based text
- TemplateGenerator: Fill-in-the-blank templates
- CodeGenerator: Simple code snippet generation
- MathGenerator: Math problem generation
- ConversationGenerator: Q&A pair generation
"""

from .simple_text_generator import (
    SyntheticTextGenerator,
    MarkovGenerator,
    GrammarGenerator,
    TemplateGenerator,
    CodeGenerator,
    MathGenerator,
    ConversationGenerator,
)

__all__ = [
    'SyntheticTextGenerator',
    'MarkovGenerator',
    'GrammarGenerator',
    'TemplateGenerator',
    'CodeGenerator',
    'MathGenerator',
    'ConversationGenerator',
]

__version__ = "1.0.0"
