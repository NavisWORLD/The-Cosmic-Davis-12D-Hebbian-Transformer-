"""
Synthetic Text Generation System for 12D Cosmic Synapse Transformer

This module provides multiple text generation strategies for creating
training data with zero cost.

Author: Cory Shane Davis
License: MIT
"""

import random
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np


class SyntheticTextGenerator:
    """
    Base class for synthetic text generation.

    All generators should inherit from this class and implement
    the generate() method.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}

    def generate(self, num_tokens: int) -> List[str]:
        """
        Generate synthetic text tokens.

        Args:
            num_tokens: Number of tokens to generate

        Returns:
            List of generated tokens
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def build_vocab(self, tokens: List[str]) -> None:
        """Build vocabulary from tokens."""
        unique_tokens = sorted(set(tokens))
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to integer IDs."""
        return [self.vocab.get(token, 0) for token in tokens]


class MarkovGenerator(SyntheticTextGenerator):
    """
    Generate text using Markov chains.

    This creates coherent-looking text by learning transition probabilities
    between n-grams.
    """

    def __init__(self, order: int = 2, seed: Optional[int] = None) -> None:
        """
        Initialize Markov generator.

        Args:
            order: Order of the Markov chain (how many previous tokens to consider)
            seed: Random seed
        """
        super().__init__(seed)
        self.order = order
        self.transitions: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
        self._initialize_corpus()

    def _initialize_corpus(self) -> None:
        """Initialize with a seed corpus."""
        seed_texts = [
            "The cosmic synapse connects all dimensions through golden ratio harmonics.",
            "In the beginning there was consciousness and light intertwined.",
            "Neural networks learn by adjusting weights through backpropagation.",
            "The universe exhibits fractal patterns at every scale.",
            "Artificial intelligence emerges from complex adaptive systems.",
            "Deep learning models transform data into meaningful representations.",
            "The golden ratio appears throughout nature and mathematics.",
            "Consciousness may arise from quantum coherence in neural microtubules.",
            "Information cannot be created or destroyed only transformed.",
            "The observer effect suggests reality is fundamentally participatory.",
        ]

        # Build transitions from seed corpus
        for text in seed_texts:
            tokens = text.split()
            for i in range(len(tokens) - self.order):
                context = tuple(tokens[i:i + self.order])
                next_token = tokens[i + self.order]
                self.transitions[context].append(next_token)

    def generate(self, num_tokens: int) -> List[str]:
        """Generate text using Markov chains."""
        if not self.transitions:
            self._initialize_corpus()

        # Start with a random context
        current_context = random.choice(list(self.transitions.keys()))
        result = list(current_context)

        for _ in range(num_tokens - self.order):
            if current_context in self.transitions:
                next_token = random.choice(self.transitions[current_context])
                result.append(next_token)
                current_context = tuple(result[-self.order:])
            else:
                # If we hit a dead end, restart with a random context
                current_context = random.choice(list(self.transitions.keys()))
                result.extend(current_context)

        return result[:num_tokens]


class GrammarGenerator(SyntheticTextGenerator):
    """Generate text using context-free grammar rules."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize grammar generator."""
        super().__init__(seed)
        self.rules = {
            'S': [
                ['NP', 'VP', '.'],
                ['NP', 'VP', 'PP', '.'],
                ['S', 'and', 'S'],
            ],
            'NP': [
                ['Det', 'N'],
                ['Det', 'Adj', 'N'],
                ['Det', 'Adj', 'Adj', 'N'],
            ],
            'VP': [
                ['V', 'NP'],
                ['V', 'PP'],
                ['V', 'NP', 'PP'],
            ],
            'PP': [
                ['Prep', 'NP'],
            ],
            'Det': [['the'], ['a'], ['an'], ['this'], ['that']],
            'N': [
                ['synapse'], ['neuron'], ['network'], ['model'], ['system'],
                ['dimension'], ['harmony'], ['consciousness'], ['pattern'], ['data']
            ],
            'Adj': [
                ['cosmic'], ['neural'], ['golden'], ['quantum'], ['fractal'],
                ['adaptive'], ['intelligent'], ['complex'], ['emergent']
            ],
            'V': [
                ['transforms'], ['connects'], ['learns'], ['emerges'], ['computes'],
                ['processes'], ['evolves'], ['generates'], ['optimizes']
            ],
            'Prep': [['in'], ['on'], ['with'], ['through'], ['via'], ['across']],
        }

    def expand(self, symbol: str) -> List[str]:
        """Recursively expand a grammar symbol."""
        if symbol not in self.rules:
            return [symbol]

        expansion = random.choice(self.rules[symbol])
        result = []
        for sym in expansion:
            result.extend(self.expand(sym))
        return result

    def generate(self, num_tokens: int) -> List[str]:
        """Generate text using grammar rules."""
        result = []
        while len(result) < num_tokens:
            sentence = self.expand('S')
            result.extend(sentence)
        return result[:num_tokens]


class TemplateGenerator(SyntheticTextGenerator):
    """Generate text by filling in templates."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize template generator."""
        super().__init__(seed)
        self.templates = [
            "The {noun} {verb} the {adjective} {noun}.",
            "In {location}, the {adjective} {noun} {verb}.",
            "Scientists discovered that {noun} can {verb}.",
            "The {adjective} {noun} represents {concept}.",
            "{number} {noun}s were {verb}ed in the {location}.",
            "By {action}ing the {noun}, we can {verb} the {noun}.",
        ]

        self.fillers = {
            'noun': ['network', 'synapse', 'model', 'system', 'pattern', 'dimension'],
            'verb': ['transforms', 'connects', 'learns', 'emerges', 'computes'],
            'adjective': ['cosmic', 'neural', 'golden', 'quantum', 'adaptive'],
            'location': ['space', 'time', 'reality', 'consciousness', 'the universe'],
            'concept': ['harmony', 'intelligence', 'complexity', 'emergence'],
            'number': ['twelve', 'seven', 'three', 'infinite', 'countless'],
            'action': ['optimiz', 'transform', 'model', 'simulat', 'train'],
        }

    def fill_template(self, template: str) -> List[str]:
        """Fill in a template with random words."""
        result = template
        for placeholder, options in self.fillers.items():
            pattern = f"{{{placeholder}}}"
            while pattern in result:
                result = result.replace(pattern, random.choice(options), 1)
        return result.split()

    def generate(self, num_tokens: int) -> List[str]:
        """Generate text by filling templates."""
        result = []
        while len(result) < num_tokens:
            template = random.choice(self.templates)
            sentence = self.fill_template(template)
            result.extend(sentence)
        return result[:num_tokens]


class CodeGenerator(SyntheticTextGenerator):
    """Generate simple code snippets."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize code generator."""
        super().__init__(seed)
        self.functions = ['compute', 'process', 'transform', 'analyze', 'optimize']
        self.variables = ['x', 'y', 'data', 'result', 'output', 'input']
        self.operations = ['+', '-', '*', '/', '**', '%']

    def generate_function(self) -> List[str]:
        """Generate a simple function."""
        func_name = random.choice(self.functions)
        param = random.choice(self.variables)
        var1 = random.choice(self.variables)
        var2 = random.choice(self.variables)
        op = random.choice(self.operations)

        code = f"def {func_name}({param}): {var1} = {param} {op} 2 return {var1}"
        return code.split()

    def generate(self, num_tokens: int) -> List[str]:
        """Generate code snippets."""
        result = []
        while len(result) < num_tokens:
            result.extend(self.generate_function())
        return result[:num_tokens]


class MathGenerator(SyntheticTextGenerator):
    """Generate math problems and solutions."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize math generator."""
        super().__init__(seed)

    def generate_problem(self) -> List[str]:
        """Generate a simple math problem."""
        num1 = random.randint(1, 100)
        num2 = random.randint(1, 100)
        operation = random.choice(['+', '-', '*'])

        if operation == '+':
            answer = num1 + num2
            op_word = 'plus'
        elif operation == '-':
            answer = num1 - num2
            op_word = 'minus'
        else:
            answer = num1 * num2
            op_word = 'times'

        problem = f"What is {num1} {op_word} {num2}? The answer is {answer}."
        return problem.split()

    def generate(self, num_tokens: int) -> List[str]:
        """Generate math problems."""
        result = []
        while len(result) < num_tokens:
            result.extend(self.generate_problem())
        return result[:num_tokens]


class ConversationGenerator(SyntheticTextGenerator):
    """Generate Q&A pairs and conversations."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize conversation generator."""
        super().__init__(seed)
        self.qa_pairs = [
            ("What is the golden ratio?", "The golden ratio is approximately 1.618."),
            ("How do neural networks learn?", "Neural networks learn by adjusting weights through backpropagation."),
            ("What is consciousness?", "Consciousness is the state of being aware of one's existence."),
            ("What are transformers?", "Transformers are attention-based neural network architectures."),
            ("How does attention work?", "Attention mechanisms allow models to focus on relevant parts of input."),
        ]

    def generate_qa(self) -> List[str]:
        """Generate a Q&A pair."""
        question, answer = random.choice(self.qa_pairs)
        conversation = f"Q: {question} A: {answer}"
        return conversation.split()

    def generate(self, num_tokens: int) -> List[str]:
        """Generate conversations."""
        result = []
        while len(result) < num_tokens:
            result.extend(self.generate_qa())
        return result[:num_tokens]
