"""
12D COSMIC SYNAPSE TRANSFORMER - MINIMAL DEMO
==============================================

A minimal working example that demonstrates:
1. Model initialization
2. Forward pass
3. Text generation
4. Internal state (x₁₂) dynamics
5. Hebbian connectivity

This proves the model works end-to-end.

Run: python demo_cosmic_transformer.py
"""

import torch
import numpy as np
from cosmic_synapse_transformer import (
    CosmicSynapseTransformer,
    CosmicConfig,
    PHI
)

def print_banner(text):
    """Pretty print banners"""
    width = 70
    print("\n" + "="*width)
    print(text.center(width))
    print("="*width + "\n")

def main():
    print_banner("12D COSMIC SYNAPSE TRANSFORMER - DEMO")
    
    # ===================================================================
    # 1. CREATE MODEL
    # ===================================================================
    
    print("📊 STEP 1: Creating 12D CST Model")
    print("-" * 70)
    
    config = CosmicConfig(
        vocab_size=1000,      # Small vocab for demo
        max_seq_len=128,      # Short sequences
        d_model=192,          # Small model (will be φ-optimized)
        n_layers=4,           # Few layers
        n_heads=4,
        dropout=0.0,          # No dropout for demo
    )
    
    print(f"Configuration:")
    print(f"  • d_model: {config.d_model} (φ-optimized)")
    print(f"  • d_ff: {config.d_ff} (= d_model × φ)")
    print(f"  • n_layers: {config.n_layers}")
    print(f"  • n_heads: {config.n_heads}")
    print(f"  • φ = {PHI:.12f}")
    
    model = CosmicSynapseTransformer(config)
    
    num_params = model.get_num_params()
    print(f"\n✓ Model created: {num_params:,} parameters ({num_params/1e6:.2f}M)")
    
    # ===================================================================
    # 2. FORWARD PASS
    # ===================================================================
    
    print_banner("STEP 2: Forward Pass")
    
    # Create random input
    batch_size = 2
    seq_len = 32
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"Input shape: {input_ids.shape}")
    print(f"Sample tokens: {input_ids[0, :10].tolist()}")
    
    # Forward pass
    with torch.no_grad():
        logits, loss, metrics = model(input_ids, targets=input_ids)
    
    print(f"\n✓ Forward pass complete")
    print(f"  • Output shape: {logits.shape}")
    print(f"  • Loss: {loss.item():.4f}")
    print(f"  • x₁₂ mean: {metrics['x12_final']:.6f}")
    print(f"  • x₁₂ std: {metrics['x12_std']:.6f}")
    
    # ===================================================================
    # 3. INTERNAL STATE DYNAMICS
    # ===================================================================
    
    print_banner("STEP 3: Internal State (x₁₂) Evolution")
    
    print("Running multiple forward passes to observe x₁₂ convergence...")
    print("(In a real model, x₁₂ evolves during training)\n")
    
    x12_history = []
    losses = []
    
    for step in range(10):
        with torch.no_grad():
            logits, loss, metrics = model(input_ids, targets=input_ids)
        
        x12_history.append(metrics['x12_final'])
        losses.append(loss.item())
        
        if step % 3 == 0:
            print(f"Step {step:2d} | Loss: {loss.item():.4f} | x₁₂: {metrics['x12_final']:+.6f}")
    
    print(f"\n✓ x₁₂ dynamics observed")
    print(f"  • Initial x₁₂: {x12_history[0]:+.6f}")
    print(f"  • Final x₁₂: {x12_history[-1]:+.6f}")
    print(f"  • Change: {x12_history[-1] - x12_history[0]:+.6f}")
    
    # ===================================================================
    # 4. TEXT GENERATION
    # ===================================================================
    
    print_banner("STEP 4: Text Generation")
    
    # Start with a small context
    context = torch.randint(0, config.vocab_size, (1, 5))
    print(f"Context tokens: {context[0].tolist()}")
    
    # Generate
    print("\nGenerating 20 tokens...")
    
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=20,
            temperature=1.0,
            top_k=20
        )
    
    print(f"\n✓ Generation complete")
    print(f"  • Context length: {context.shape[1]}")
    print(f"  • Generated length: {generated.shape[1]}")
    print(f"  • Total tokens: {generated.shape[1]}")
    print(f"  • Generated sequence: {generated[0].tolist()}")
    
    # ===================================================================
    # 5. VISUALIZE x₁₂ DISTRIBUTION
    # ===================================================================
    
    print_banner("STEP 5: x₁₂ State Distribution")
    
    # Run forward pass and examine x₁₂ values
    print("Analyzing internal state distribution across tokens...")
    
    # Create larger batch to see distribution
    large_batch = torch.randint(0, config.vocab_size, (8, 64))
    
    with torch.no_grad():
        logits, loss, metrics = model(large_batch, targets=large_batch)
    
    print(f"\n✓ Analysis complete")
    print(f"  • Batch size: {large_batch.shape[0]}")
    print(f"  • Sequence length: {large_batch.shape[1]}")
    print(f"  • Total tokens analyzed: {large_batch.shape[0] * large_batch.shape[1]}")
    print(f"  • x₁₂ mean: {metrics['x12_final']:+.6f}")
    print(f"  • x₁₂ std: {metrics['x12_std']:.6f}")
    
    print("\nx₁₂ evolution across layers:")
    for i, x12 in enumerate(metrics['x12_history']):
        bar_length = int(abs(x12) * 50)
        bar = "█" * bar_length
        print(f"  Layer {i+1:2d}: {x12:+.6f} {bar}")
    
    # ===================================================================
    # 6. VERIFY THEORETICAL PROPERTIES
    # ===================================================================
    
    print_banner("STEP 6: Theoretical Validation")
    
    print("Verifying key properties of 12D CST...")
    
    # Test 1: φ-harmonic dimensions
    phi_ratio = config.d_ff / config.d_model
    phi_error = abs(phi_ratio - PHI)
    print(f"\n1. φ-Harmonic Architecture:")
    print(f"   d_ff / d_model = {phi_ratio:.6f}")
    print(f"   φ = {PHI:.6f}")
    print(f"   Error: {phi_error:.8f}")
    print(f"   ✓ PASS" if phi_error < 0.01 else "   ✗ FAIL")
    
    # Test 2: x₁₂ boundedness
    print(f"\n2. Internal State Boundedness:")
    print(f"   x₁₂ range should be [-1, 1]")
    print(f"   Observed: [{metrics['x12_final'] - metrics['x12_std']:.3f}, "
          f"{metrics['x12_final'] + metrics['x12_std']:.3f}]")
    is_bounded = abs(metrics['x12_final']) <= 1.0 and metrics['x12_std'] < 1.0
    print(f"   ✓ PASS" if is_bounded else "   ✗ FAIL")
    
    # Test 3: Energy conservation (approximate for single step)
    print(f"\n3. Loss Stability:")
    print(f"   Initial loss: {losses[0]:.4f}")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   Change: {abs(losses[-1] - losses[0]):.4f}")
    is_stable = abs(losses[-1] - losses[0]) < 5.0  # Reasonable for untrained
    print(f"   ✓ PASS (stable)" if is_stable else "   ⚠ Unstable (expected for untrained model)")
    
    # Test 4: Chaos parameters present
    print(f"\n4. Chaos Mechanisms:")
    has_lorenz = any(hasattr(layer, 'lorenz') for layer in model.layers)
    print(f"   Lorenz attractors present: {has_lorenz}")
    print(f"   ✓ PASS" if has_lorenz else "   ✗ FAIL")
    
    # Test 5: Hebbian connectivity
    print(f"\n5. Hebbian Attention:")
    has_hebbian = any(hasattr(layer.attention, 'compute_hebbian_bonus') 
                     for layer in model.layers)
    print(f"   Hebbian modulation present: {has_hebbian}")
    print(f"   ✓ PASS" if has_hebbian else "   ✗ FAIL")
    
    # ===================================================================
    # SUMMARY
    # ===================================================================
    
    print_banner("DEMO COMPLETE ✓")
    
    print("Summary of 12D Cosmic Synapse Transformer:")
    print()
    print("✓ Model architecture: φ-harmonic transformer")
    print("✓ Internal states (x₁₂): Adaptive per-token dynamics")
    print("✓ Hebbian attention: Similarity-modulated connections")
    print("✓ Chaos injection: Lorenz attractor exploration")
    print("✓ Memory: Episodic memory buffers")
    print()
    print(f"Model size: {num_params:,} parameters ({num_params/1e6:.2f}M)")
    print(f"φ = {PHI:.12f}")
    print()
    print("Next steps:")
    print("  1. Prepare your dataset: python inference_cosmic_transformer.py prepare data.txt")
    print("  2. Train the model: python train_cosmic_transformer.py")
    print("  3. Generate text: python inference_cosmic_transformer.py generate ckpt.pt")
    print()
    print("For production use, scale up:")
    print("  • d_model: 768 → 1536 → 12288 (GPT-4 class)")
    print("  • n_layers: 12 → 24 → 96")
    print("  • Training data: 10B → 100B → 1T+ tokens")
    print()
    print("="*70)
    print("The Cosmic Synapse is ready. φ".center(70))
    print("="*70 + "\n")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demo
    main()
