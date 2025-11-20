#!/usr/bin/env python
"""
Synthetic Data Generation for 12D Cosmic Synapse Transformer

This script generates synthetic training data with $0 cost, creating realistic
text using multiple generation strategies including Markov chains, grammar rules,
templates, code snippets, math problems, and conversations.

The generated data includes subtle φ-harmonic patterns and is saved in binary
format compatible with the TextDataset class.

Usage:
    python generate_synthetic_data.py --num-tokens 10000000 --output-dir data

Author: Cory Shane Davis
License: MIT
"""

import argparse
import os
import struct
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import random

from datasets.simple_text_generator import (
    MarkovGenerator,
    GrammarGenerator,
    TemplateGenerator,
    CodeGenerator,
    MathGenerator,
    ConversationGenerator,
)


class SyntheticDataGenerator:
    """
    Orchestrates multiple text generators to create diverse synthetic data.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Initialize all generators
        self.generators = {
            'markov': MarkovGenerator(order=2, seed=seed),
            'grammar': GrammarGenerator(seed=seed + 1),
            'template': TemplateGenerator(seed=seed + 2),
            'code': CodeGenerator(seed=seed + 3),
            'math': MathGenerator(seed=seed + 4),
            'conversation': ConversationGenerator(seed=seed + 5),
        }

        # Weights for each generator (can be adjusted)
        self.generator_weights = {
            'markov': 0.3,
            'grammar': 0.25,
            'template': 0.2,
            'code': 0.1,
            'math': 0.1,
            'conversation': 0.05,
        }

        self.vocab: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.vocab_size = 4

    def add_phi_harmonic_patterns(self, tokens: List[str]) -> List[str]:
        """
        Add subtle φ-harmonic patterns to the generated text.

        This inserts golden ratio related tokens at φ-harmonic intervals
        to create hidden structure in the data.
        """
        phi = 1.618033988749895
        phi_tokens = ['golden', 'harmony', 'ratio', 'phi', 'cosmic']

        result = tokens.copy()
        length = len(tokens)

        # Insert φ-related tokens at φ-harmonic positions
        positions = []
        current = int(length / phi)
        while current < length:
            positions.append(current)
            current = int(current * phi)
            if current >= length:
                break

        for pos in positions[::-1]:  # Insert in reverse to preserve positions
            if pos < len(result):
                result.insert(pos, random.choice(phi_tokens))

        return result

    def generate_tokens(self, num_tokens: int, add_phi_patterns: bool = True) -> List[str]:
        """
        Generate synthetic tokens using a mix of all generators.

        Args:
            num_tokens: Number of tokens to generate
            add_phi_patterns: Whether to add φ-harmonic patterns

        Returns:
            List of generated tokens
        """
        all_tokens = []

        with tqdm(total=num_tokens, desc="Generating tokens") as pbar:
            while len(all_tokens) < num_tokens:
                # Select generator based on weights
                generator_name = random.choices(
                    list(self.generators.keys()),
                    weights=list(self.generator_weights.values()),
                    k=1
                )[0]

                generator = self.generators[generator_name]

                # Generate a chunk of tokens
                chunk_size = min(100, num_tokens - len(all_tokens))
                chunk = generator.generate(chunk_size)

                all_tokens.extend(chunk)
                pbar.update(len(chunk))

        # Truncate to exact size
        all_tokens = all_tokens[:num_tokens]

        # Add φ-harmonic patterns
        if add_phi_patterns:
            all_tokens = self.add_phi_harmonic_patterns(all_tokens)

        return all_tokens

    def build_vocabulary(self, tokens: List[str], max_vocab_size: int = 10000) -> None:
        """
        Build vocabulary from tokens.

        Args:
            tokens: List of tokens
            max_vocab_size: Maximum vocabulary size
        """
        from collections import Counter

        print(f"Building vocabulary from {len(tokens)} tokens...")

        # Count token frequencies
        token_counts = Counter(tokens)

        # Get most common tokens
        most_common = token_counts.most_common(max_vocab_size - len(self.vocab))

        # Add to vocabulary
        for token, count in most_common:
            if token not in self.vocab:
                self.vocab[token] = self.vocab_size
                self.vocab_size += 1

        print(f"Vocabulary size: {self.vocab_size}")

    def tokens_to_ids(self, tokens: List[str]) -> np.ndarray:
        """
        Convert tokens to integer IDs.

        Args:
            tokens: List of tokens

        Returns:
            Numpy array of token IDs
        """
        ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        return np.array(ids, dtype=np.uint16)

    def save_binary(self, ids: np.ndarray, output_path: str) -> None:
        """
        Save token IDs in binary format.

        Args:
            ids: Array of token IDs
            output_path: Output file path
        """
        print(f"Saving {len(ids)} tokens to {output_path}...")

        with open(output_path, 'wb') as f:
            # Write header: magic number and vocab size
            f.write(struct.pack('I', 0x12D5C7))  # Magic number for 12D CST
            f.write(struct.pack('I', self.vocab_size))

            # Write token IDs
            ids.tofile(f)

        print(f"Saved {len(ids)} tokens ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")

    def save_vocabulary(self, output_path: str) -> None:
        """Save vocabulary to text file."""
        vocab_path = os.path.join(os.path.dirname(output_path), 'vocab.txt')
        print(f"Saving vocabulary to {vocab_path}...")

        with open(vocab_path, 'w') as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")


def main():
    """Main function to generate synthetic data."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic training data for 12D Cosmic Synapse Transformer'
    )
    parser.add_argument(
        '--num-tokens',
        type=int,
        default=10_000_000,
        help='Total number of tokens to generate (default: 10M)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory (default: data)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.9,
        help='Proportion of data for training (default: 0.9)'
    )
    parser.add_argument(
        '--max-vocab-size',
        type=int,
        default=10000,
        help='Maximum vocabulary size (default: 10000)'
    )
    parser.add_argument(
        '--no-phi-patterns',
        action='store_true',
        help='Disable φ-harmonic pattern insertion'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("12D COSMIC SYNAPSE TRANSFORMER - SYNTHETIC DATA GENERATION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Total tokens: {args.num_tokens:,}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Random seed: {args.seed}")
    print(f"  Train/Val split: {args.train_split:.1%} / {1-args.train_split:.1%}")
    print(f"  Max vocab size: {args.max_vocab_size:,}")
    print(f"  φ-harmonic patterns: {not args.no_phi_patterns}")
    print("=" * 80)

    # Initialize generator
    generator = SyntheticDataGenerator(seed=args.seed)

    # Generate tokens
    print("\n🌌 Generating synthetic tokens...")
    tokens = generator.generate_tokens(
        args.num_tokens,
        add_phi_patterns=not args.no_phi_patterns
    )

    # Build vocabulary
    print("\n📚 Building vocabulary...")
    generator.build_vocabulary(tokens, max_vocab_size=args.max_vocab_size)

    # Convert to IDs
    print("\n🔢 Converting tokens to IDs...")
    token_ids = generator.tokens_to_ids(tokens)

    # Split into train/val
    split_idx = int(len(token_ids) * args.train_split)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    print(f"\n📊 Data split:")
    print(f"  Training: {len(train_ids):,} tokens")
    print(f"  Validation: {len(val_ids):,} tokens")

    # Save data
    print("\n💾 Saving data...")
    train_path = output_dir / 'train.bin'
    val_path = output_dir / 'val.bin'

    generator.save_binary(train_ids, str(train_path))
    generator.save_binary(val_ids, str(val_path))
    generator.save_vocabulary(str(train_path))

    # Validation
    print("\n✅ Validating output...")
    assert train_path.exists(), "Train file not created"
    assert val_path.exists(), "Val file not created"

    train_size_mb = train_path.stat().st_size / 1024 / 1024
    val_size_mb = val_path.stat().st_size / 1024 / 1024

    print(f"  ✓ Train file: {train_size_mb:.2f} MB")
    print(f"  ✓ Val file: {val_size_mb:.2f} MB")

    print("\n" + "=" * 80)
    print("✨ SYNTHETIC DATA GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nYou can now train the model with:")
    print(f"  python train_cosmic_transformer.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
