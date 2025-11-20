"""
Data Generation and Loading Tests for 12D Cosmic Synapse Transformer

Tests data components including:
- Synthetic data generation
- Text generators
- Dataset loading
- Vocabulary building

Author: Cory Shane Davis
License: MIT
"""

import pytest
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.simple_text_generator import (
    MarkovGenerator,
    GrammarGenerator,
    TemplateGenerator,
    CodeGenerator,
    MathGenerator,
    ConversationGenerator,
)
from generate_synthetic_data import SyntheticDataGenerator


class TestTextGenerators:
    """Test individual text generators."""

    def test_markov_generator(self):
        """Test Markov chain generator."""
        gen = MarkovGenerator(order=2, seed=42)
        tokens = gen.generate(100)

        assert len(tokens) == 100
        assert all(isinstance(t, str) for t in tokens)

    def test_grammar_generator(self):
        """Test grammar-based generator."""
        gen = GrammarGenerator(seed=42)
        tokens = gen.generate(100)

        assert len(tokens) >= 100  # Might be slightly more due to sentence completion
        assert all(isinstance(t, str) for t in tokens)

    def test_template_generator(self):
        """Test template-based generator."""
        gen = TemplateGenerator(seed=42)
        tokens = gen.generate(100)

        assert len(tokens) == 100
        assert all(isinstance(t, str) for t in tokens)

    def test_code_generator(self):
        """Test code snippet generator."""
        gen = CodeGenerator(seed=42)
        tokens = gen.generate(100)

        assert len(tokens) == 100
        assert all(isinstance(t, str) for t in tokens)

    def test_math_generator(self):
        """Test math problem generator."""
        gen = MathGenerator(seed=42)
        tokens = gen.generate(100)

        assert len(tokens) == 100
        assert 'answer' in ' '.join(tokens).lower()

    def test_conversation_generator(self):
        """Test conversation generator."""
        gen = ConversationGenerator(seed=42)
        tokens = gen.generate(100)

        assert len(tokens) == 100
        # Should contain Q: and A: markers
        text = ' '.join(tokens)
        assert 'Q:' in text or 'A:' in text

    def test_generator_reproducibility(self):
        """Test that generators are reproducible with same seed."""
        gen1 = MarkovGenerator(seed=42)
        gen2 = MarkovGenerator(seed=42)

        tokens1 = gen1.generate(50)
        tokens2 = gen2.generate(50)

        assert tokens1 == tokens2


class TestSyntheticDataGenerator:
    """Test synthetic data generation system."""

    def test_basic_generation(self):
        """Test basic token generation."""
        gen = SyntheticDataGenerator(seed=42)
        tokens = gen.generate_tokens(1000, add_phi_patterns=False)

        assert len(tokens) == 1000
        assert all(isinstance(t, str) for t in tokens)

    def test_phi_harmonic_patterns(self):
        """Test φ-harmonic pattern insertion."""
        gen = SyntheticDataGenerator(seed=42)

        # Generate without phi patterns
        tokens_no_phi = gen.generate_tokens(1000, add_phi_patterns=False)

        # Generate with phi patterns
        tokens_with_phi = gen.generate_tokens(1000, add_phi_patterns=True)

        # With phi patterns should be slightly longer due to insertions
        assert len(tokens_with_phi) > len(tokens_no_phi)

        # Should contain phi-related words
        phi_text = ' '.join(tokens_with_phi).lower()
        phi_words = ['golden', 'harmony', 'ratio', 'phi', 'cosmic']
        assert any(word in phi_text for word in phi_words)

    def test_vocabulary_building(self):
        """Test vocabulary construction."""
        gen = SyntheticDataGenerator(seed=42)
        tokens = gen.generate_tokens(1000, add_phi_patterns=False)

        gen.build_vocabulary(tokens, max_vocab_size=500)

        # Vocab should be built
        assert len(gen.vocab) > 0
        assert len(gen.vocab) <= 500 + 4  # +4 for special tokens

        # Special tokens should be present
        assert '<PAD>' in gen.vocab
        assert '<UNK>' in gen.vocab
        assert '<BOS>' in gen.vocab
        assert '<EOS>' in gen.vocab

    def test_token_to_id_conversion(self):
        """Test converting tokens to IDs."""
        gen = SyntheticDataGenerator(seed=42)
        tokens = gen.generate_tokens(100, add_phi_patterns=False)

        gen.build_vocabulary(tokens, max_vocab_size=100)
        ids = gen.tokens_to_ids(tokens)

        # Should have same length
        assert len(ids) == len(tokens)

        # All IDs should be valid
        assert all(0 <= id < gen.vocab_size for id in ids)


class TestDataSaving:
    """Test saving and loading data files."""

    def test_save_binary_data(self):
        """Test saving data in binary format."""
        gen = SyntheticDataGenerator(seed=42)
        tokens = gen.generate_tokens(1000, add_phi_patterns=False)
        gen.build_vocabulary(tokens, max_vocab_size=100)
        ids = gen.tokens_to_ids(tokens)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.bin'
            gen.save_binary(ids, str(output_path))

            # File should exist
            assert output_path.exists()

            # File should have content
            assert output_path.stat().st_size > 0

    def test_save_vocabulary(self):
        """Test saving vocabulary file."""
        gen = SyntheticDataGenerator(seed=42)
        tokens = gen.generate_tokens(100, add_phi_patterns=False)
        gen.build_vocabulary(tokens, max_vocab_size=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.bin'
            gen.save_vocabulary(str(output_path))

            vocab_path = Path(tmpdir) / 'vocab.txt'

            # Vocab file should exist
            assert vocab_path.exists()

            # Should contain tokens
            with open(vocab_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 0

                # Each line should be "token\tid"
                for line in lines[:5]:  # Check first few
                    parts = line.strip().split('\t')
                    assert len(parts) == 2


@pytest.mark.slow
class TestFullDataGeneration:
    """Test complete data generation pipeline."""

    def test_generate_train_val_split(self):
        """Test generating training and validation data."""
        gen = SyntheticDataGenerator(seed=42)

        # Generate tokens
        num_tokens = 10000
        tokens = gen.generate_tokens(num_tokens, add_phi_patterns=True)

        # Build vocab
        gen.build_vocabulary(tokens, max_vocab_size=500)

        # Convert to IDs
        ids = gen.tokens_to_ids(tokens)

        # Split
        split_idx = int(len(ids) * 0.9)
        train_ids = ids[:split_idx]
        val_ids = ids[split_idx:]

        # Check split
        assert len(train_ids) + len(val_ids) == len(ids)
        assert len(train_ids) > len(val_ids)

        # Save both
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / 'train.bin'
            val_path = Path(tmpdir) / 'val.bin'

            gen.save_binary(train_ids, str(train_path))
            gen.save_binary(val_ids, str(val_path))

            assert train_path.exists()
            assert val_path.exists()
            assert train_path.stat().st_size > val_path.stat().st_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
