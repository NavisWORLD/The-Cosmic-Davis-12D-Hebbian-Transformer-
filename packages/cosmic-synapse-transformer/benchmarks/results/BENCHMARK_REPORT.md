# 🔬 BENCHMARK RESULTS: 12D Cosmic Synapse Transformer vs Vanilla Transformer

**Date**: 2025-11-20 03:52:46
**Device**: CPU
**PyTorch Version**: 2.9.1+cu128

---

## 📊 Model Configurations

| Metric | Vanilla Transformer | 12D CST | Difference |
|--------|-------------------|---------|------------|
| **Parameters** | 1,996,416 | 1,347,820 | 648,596 |
| **d_model** | 192 | 196 | - |
| **d_ff** | 768 | 321 | φ-harmonic |
| **Layers** | 4 | 4 | - |
| **Heads** | 4 | 4 | - |

---

## 🎯 Training Results

### Final Metrics:

| Metric | Vanilla | 12D CST | Winner |
|--------|---------|---------|--------|
| **Train Loss** | 1.5041 | 1.6187 | 🏆 Vanilla |
| **Val Loss** | 1.4362 | 1.4445 | 🏆 Vanilla |
| **Avg Speed** | 9088 tok/s | 8461 tok/s | 🏆 Vanilla |

### Convergence:

- **Vanilla**: 6.9574 → 1.5041 (Δ 5.4533)
- **12D CST**: 6.9262 → 1.6187 (Δ 5.3075)

### Perplexity:

- **Vanilla Val**: 4.20
- **12D CST Val**: 4.24

---

## 🏆 Winner Analysis

✅ **Vanilla** achieves lower training loss
✅ **Vanilla** achieves lower validation loss
✅ **Vanilla** is 1.1x faster

**Overall Winner**: 🏆 VANILLA TRANSFORMER

---

## 📈 Raw Data

Benchmark data saved to: `benchmark_results.json`
