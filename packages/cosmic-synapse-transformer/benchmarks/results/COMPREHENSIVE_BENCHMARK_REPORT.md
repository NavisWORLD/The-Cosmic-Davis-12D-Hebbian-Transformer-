# 🔬 COMPREHENSIVE REAL-WORLD-SCALE BENCHMARK RESULTS

**Date**: 2025-11-20 04:42:25
**Dataset**: Large-scale synthetic (1M tokens)
**Device**: CPU
**PyTorch**: 2.9.1+cu128

---

## 📊 EXPERIMENT 1: Extended Training (2000 iterations)

**Purpose**: Test convergence over longer training runs


| Metric | Vanilla | 12D CST | Winner |
|--------|---------|---------|--------|
| **Parameters** | 2,764,416 | 2,131,820 | - |
| **Final Train Loss** | 1.3991 | 1.5857 | Vanilla |
| **Final Val Loss** | 1.2957 | 1.1295 | 12D CST |
| **Perplexity** | 3.65 | 3.09 | 12D CST |
| **Avg Speed** | 7524 tok/s | 6662 tok/s | - |

**Convergence**:
- Vanilla: 8.5651 → 1.3991 (Δ 7.1661)
- 12D CST: 8.5283 → 1.5857 (Δ 6.9425)

---

## 📊 EXPERIMENT 2: Parameter-Matched Comparison

**Purpose**: Fair architectural comparison with equal parameters


| Metric | Vanilla | 12D CST | Winner |
|--------|---------|---------|--------|
| **Parameters** | 2,764,416 | 4,622,252 | Unmatched |
| **Final Train Loss** | 1.4008 | 1.3000 | 🏆 12D CST |
| **Final Val Loss** | 1.4603 | 1.4241 | 🏆 12D CST |
| **Perplexity** | 4.31 | 4.15 | 🏆 12D CST |

---

## 📊 EXPERIMENT 3: Scaling Analysis

**Purpose**: Test how architectures scale across model sizes


### TINY Models

| Metric | Vanilla | 12D CST | Winner |
|--------|---------|---------|--------|
| Parameters | 1,053,184 | 843,666 | - |
| Train Loss | 1.4976 | 1.7503 | Vanilla |
| Val Loss | 1.5138 | 1.5833 | Vanilla |
| Perplexity | 4.54 | 4.87 | Vanilla |


### SMALL Models

| Metric | Vanilla | 12D CST | Winner |
|--------|---------|---------|--------|
| Parameters | 2,764,416 | 2,131,820 | - |
| Train Loss | 1.3086 | 1.6064 | Vanilla |
| Val Loss | 1.5522 | 1.1834 | 12D CST |
| Perplexity | 4.72 | 3.27 | 12D CST |


### MEDIUM Models

| Metric | Vanilla | 12D CST | Winner |
|--------|---------|---------|--------|
| Parameters | 6,051,840 | 6,112,578 | - |
| Train Loss | 1.4065 | 2.6623 | Vanilla |
| Val Loss | 1.2977 | 2.6144 | Vanilla |
| Perplexity | 3.66 | 13.66 | Vanilla |


---

## 🏆 OVERALL CONCLUSIONS

[Analysis will be provided in detailed discussion]

---

Raw data saved to: `comprehensive_benchmark_results.json`
