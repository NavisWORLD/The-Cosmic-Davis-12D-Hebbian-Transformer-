"""
Evaluation Metrics for 12D Cosmic Synapse Transformer

Functions for computing perplexity, x12 convergence, and generation quality.

Author: Cory Shane Davis
License: MIT
"""

import torch
import numpy as np
from typing import List, Dict
import math


def compute_perplexity(model: torch.nn.Module, dataset: torch.Tensor) -> float:
    """
    Compute perplexity on a dataset.

    Args:
        model: The transformer model
        dataset: Tensor of token IDs [num_samples, seq_len]

    Returns:
        Perplexity value
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataset:
            if batch.dim() == 1:
                batch = batch.unsqueeze(0)

            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits, loss = model(inputs, targets)

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


def compute_x12_convergence(x12_history: List[torch.Tensor]) -> Dict[str, float]:
    """
    Measure how well x12 converges over time.

    Args:
        x12_history: List of x12 tensors from different time steps

    Returns:
        Dictionary with convergence metrics
    """
    if len(x12_history) < 2:
        return {'convergence_rate': 0.0, 'final_variance': 0.0}

    # Convert to numpy array
    x12_array = np.array([x.cpu().detach().numpy() for x in x12_history])

    # Compute changes between consecutive steps
    changes = np.diff(x12_array, axis=0)
    change_magnitudes = np.abs(changes).mean(axis=1)

    # Convergence rate: how much the change decreases over time
    if len(change_magnitudes) > 10:
        early_change = change_magnitudes[:len(change_magnitudes)//3].mean()
        late_change = change_magnitudes[-len(change_magnitudes)//3:].mean()

        if early_change > 0:
            convergence_rate = (early_change - late_change) / early_change
        else:
            convergence_rate = 0.0
    else:
        convergence_rate = 0.0

    # Final variance: how stable are the final states
    final_x12 = x12_array[-min(10, len(x12_array)):]
    final_variance = np.var(final_x12)

    return {
        'convergence_rate': float(convergence_rate),
        'final_variance': float(final_variance),
        'mean_change': float(change_magnitudes.mean()),
        'final_x12_mean': float(np.abs(x12_array[-1]).mean()),
    }


def evaluate_generation_quality(
    generated_tokens: List[List[int]],
    vocab_size: int
) -> Dict[str, float]:
    """
    Evaluate quality of generated text.

    Args:
        generated_tokens: List of generated token sequences
        vocab_size: Size of vocabulary

    Returns:
        Dictionary with quality metrics
    """
    if not generated_tokens:
        return {}

    # Flatten all tokens
    all_tokens = [token for seq in generated_tokens for token in seq]

    # Unique tokens (vocabulary diversity)
    unique_tokens = len(set(all_tokens))
    vocab_usage = unique_tokens / vocab_size

    # Token distribution entropy
    token_counts = np.bincount(all_tokens, minlength=vocab_size)
    token_probs = token_counts / token_counts.sum()
    token_probs = token_probs[token_probs > 0]  # Remove zeros
    entropy = -np.sum(token_probs * np.log(token_probs))

    # Repetition rate
    bigrams = []
    for seq in generated_tokens:
        for i in range(len(seq) - 1):
            bigrams.append((seq[i], seq[i+1]))

    unique_bigrams = len(set(bigrams))
    total_bigrams = len(bigrams)
    repetition_rate = 1.0 - (unique_bigrams / total_bigrams if total_bigrams > 0 else 0.0)

    # Average sequence length
    avg_length = np.mean([len(seq) for seq in generated_tokens])

    return {
        'vocab_usage': float(vocab_usage),
        'token_entropy': float(entropy),
        'repetition_rate': float(repetition_rate),
        'unique_bigrams': unique_bigrams,
        'total_bigrams': total_bigrams,
        'avg_sequence_length': float(avg_length),
    }


def compute_hebbian_strength(omega_matrix: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about Hebbian connectivity.

    Args:
        omega_matrix: Hebbian connectivity matrix

    Returns:
        Dictionary with Hebbian metrics
    """
    omega_np = omega_matrix.cpu().detach().numpy()

    return {
        'mean_strength': float(np.mean(omega_np)),
        'max_strength': float(np.max(omega_np)),
        'min_strength': float(np.min(omega_np)),
        'std_strength': float(np.std(omega_np)),
        'sparsity': float((np.abs(omega_np) < 0.01).mean()),
    }


def compute_attention_entropy(attention_weights: torch.Tensor) -> float:
    """
    Compute entropy of attention distribution.

    Args:
        attention_weights: Attention weights [n_heads, seq_len, seq_len]

    Returns:
        Average entropy across heads
    """
    attn_np = attention_weights.cpu().detach().numpy()

    entropies = []
    for head in range(attn_np.shape[0]):
        for query in range(attn_np.shape[1]):
            probs = attn_np[head, query]
            probs = probs[probs > 0]
            if len(probs) > 0:
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)

    return float(np.mean(entropies)) if entropies else 0.0


def benchmark_inference_speed(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    num_tokens: int = 100,
    num_trials: int = 10
) -> Dict[str, float]:
    """
    Benchmark inference speed.

    Args:
        model: The transformer model
        prompt: Input prompt tensor
        num_tokens: Number of tokens to generate
        num_trials: Number of trials to average

    Returns:
        Dictionary with timing metrics
    """
    import time

    model.eval()
    device = next(model.parameters()).device

    times = []
    with torch.no_grad():
        # Warmup
        _ = model.generate(prompt, max_new_tokens=10)

        # Benchmark
        for _ in range(num_trials):
            start = time.time()
            _ = model.generate(prompt, max_new_tokens=num_tokens)
            end = time.time()
            times.append(end - start)

    times = np.array(times)

    return {
        'mean_time': float(times.mean()),
        'std_time': float(times.std()),
        'min_time': float(times.min()),
        'max_time': float(times.max()),
        'tokens_per_second': float(num_tokens / times.mean()),
    }
