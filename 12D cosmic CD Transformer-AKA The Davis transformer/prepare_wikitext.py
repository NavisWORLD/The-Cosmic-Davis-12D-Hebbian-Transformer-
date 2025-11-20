"""
REAL-WORLD BENCHMARK: WikiText-2 Dataset
=========================================

Download, prepare, and benchmark both models on WikiText-2,
a standard language modeling benchmark from real Wikipedia text.

WikiText-2: ~2M tokens from Wikipedia articles
"""

import os
import torch
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
from typing import List, Tuple
from collections import Counter

def download_wikitext2(data_dir: str = "data_wikitext") -> None:
    """Download WikiText-2 dataset."""

    Path(data_dir).mkdir(exist_ok=True)

    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
    zip_path = os.path.join(data_dir, "wikitext-2-raw-v1.zip")

    if not os.path.exists(zip_path):
        print(f"[DOWNLOAD] Downloading WikiText-2 from {url}")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[DOWNLOAD] Downloaded to {zip_path}")
    else:
        print(f"[DOWNLOAD] WikiText-2 already downloaded")

    # Extract
    extract_dir = os.path.join(data_dir, "wikitext-2-raw")
    if not os.path.exists(extract_dir):
        print(f"[EXTRACT] Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"[EXTRACT] Extracted to {extract_dir}")
    else:
        print(f"[EXTRACT] Already extracted")

    return extract_dir

def load_text(filepath: str) -> str:
    """Load text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def simple_tokenize(text: str) -> List[str]:
    """Simple word-level tokenization."""
    # Replace newlines with space
    text = text.replace('\n', ' ')
    # Split on whitespace and punctuation
    import re
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    return tokens

def build_vocab(tokens: List[str], vocab_size: int = 10000) -> Tuple[dict, dict]:
    """Build vocabulary from tokens."""
    counter = Counter(tokens)
    # Keep most common tokens
    most_common = counter.most_common(vocab_size - 2)  # Reserve for <unk> and <pad>

    vocab = {'<pad>': 0, '<unk>': 1}
    for i, (token, _) in enumerate(most_common):
        vocab[token] = i + 2

    reverse_vocab = {v: k for k, v in vocab.items()}

    print(f"[VOCAB] Built vocabulary with {len(vocab):,} tokens")
    return vocab, reverse_vocab

def tokens_to_ids(tokens: List[str], vocab: dict) -> np.ndarray:
    """Convert tokens to IDs."""
    return np.array([vocab.get(t, vocab['<unk>']) for t in tokens], dtype=np.uint16)

def prepare_wikitext2(vocab_size: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download and prepare WikiText-2 dataset."""

    print("="*80)
    print("PREPARING WIKITEXT-2 DATASET")
    print("="*80)

    # Download
    extract_dir = download_wikitext2()

    # Load files
    train_path = os.path.join(extract_dir, "wikitext-2-raw", "wiki.train.raw")
    valid_path = os.path.join(extract_dir, "wikitext-2-raw", "wiki.valid.raw")
    test_path = os.path.join(extract_dir, "wikitext-2-raw", "wiki.test.raw")

    print(f"\n[LOAD] Loading text files...")
    train_text = load_text(train_path)
    valid_text = load_text(valid_path)
    test_text = load_text(test_path)

    print(f"[LOAD] Train: {len(train_text):,} chars")
    print(f"[LOAD] Valid: {len(valid_text):,} chars")
    print(f"[LOAD] Test: {len(test_text):,} chars")

    # Tokenize
    print(f"\n[TOKENIZE] Tokenizing...")
    train_tokens = simple_tokenize(train_text)
    valid_tokens = simple_tokenize(valid_text)
    test_tokens = simple_tokenize(test_text)

    print(f"[TOKENIZE] Train: {len(train_tokens):,} tokens")
    print(f"[TOKENIZE] Valid: {len(valid_tokens):,} tokens")
    print(f"[TOKENIZE] Test: {len(test_tokens):,} tokens")

    # Build vocabulary from train
    vocab, reverse_vocab = build_vocab(train_tokens, vocab_size)

    # Convert to IDs
    print(f"\n[CONVERT] Converting to IDs...")
    train_ids = tokens_to_ids(train_tokens, vocab)
    valid_ids = tokens_to_ids(valid_tokens, vocab)
    test_ids = tokens_to_ids(test_tokens, vocab)

    print(f"[CONVERT] Train: {len(train_ids):,} IDs")
    print(f"[CONVERT] Valid: {len(valid_ids):,} IDs")
    print(f"[CONVERT] Test: {len(test_ids):,} IDs")

    # Save to binary
    data_dir = Path("data_wikitext")
    data_dir.mkdir(exist_ok=True)

    train_ids.tofile(data_dir / "train.bin")
    valid_ids.tofile(data_dir / "val.bin")
    test_ids.tofile(data_dir / "test.bin")

    print(f"\n[SAVE] Saved binary files to {data_dir}/")
    print(f"[SAVE] train.bin: {(data_dir / 'train.bin').stat().st_size / 1024:.1f} KB")
    print(f"[SAVE] val.bin: {(data_dir / 'val.bin').stat().st_size / 1024:.1f} KB")
    print(f"[SAVE] test.bin: {(data_dir / 'test.bin').stat().st_size / 1024:.1f} KB")

    # Save vocab
    import json
    with open(data_dir / "vocab.json", "w") as f:
        json.dump(vocab, f)

    print(f"\n✅ WikiText-2 preparation complete!")
    print(f"   Vocabulary size: {len(vocab):,}")
    print(f"   Train tokens: {len(train_ids):,}")
    print(f"   Valid tokens: {len(valid_ids):,}")

    return train_ids, valid_ids, test_ids

if __name__ == "__main__":
    prepare_wikitext2(vocab_size=10000)
