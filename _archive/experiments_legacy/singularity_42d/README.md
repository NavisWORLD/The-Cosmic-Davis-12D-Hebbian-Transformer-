# EXPERIMENTS FOLDER - FULLY CONFIGURED ✅

This folder contains **experimental** versions of the training scripts.

## 📁 Structure

```
experiments/
├── singularity_42d/           # The main singularity experiment
│   ├── autonomous_study.py    # ✅ UPDATED with multimodal support
│   ├── hyper_cosmic_model.py  # ✅ UPDATED 42D model
│   ├── chat_with_model.py     # Chat interfaces
│   ├── teacher_eval.py        # Evaluation scripts
│   └── train_*.py             # Training variants
└── ...
```

## 🔗 Dependencies

This folder is **fully configured** and has access to:
- ✅ `packages/cosmic-synapse-transformer/` (Core 12D model)
- ✅ `audio_12d/` (Multimodal sensory system - copied to root)
- ✅ All multimodal fusion components

## 🚀 How to Run

**Option 1: Use the Main Launcher (Recommended)**
```bash
START_COSMIC_DAVIS.bat
```
This uses the `research_42d` version which is the PRIMARY development location.

**Option 2: Run from Experiments Folder Directly**
```bash
cd experiments/singularity_42d
set PYTHONPATH=%CD%\..\..\packages\cosmic-synapse-transformer
python autonomous_study.py
```

## ⚠️ Important Notes

1. **Two Locations Exist:**
   - `research_42d/experiments/singularity_42d/` ← **PRIMARY** (actively developed)
   - `experiments/singularity_42d/` ← **MIRROR** (synced copy for testing)

2. **Automatic Syncing:**
   - Files from `research_42d` have been copied here
   - If you make changes here, they won't auto-sync back

3. **Multimodal Support:**
   - ✅ This version now has **full multimodal integration**
   - ✅ Emotion detection, thought generation, token streaming
   - ✅ All features from the main research branch

## 🧪 Verification

Run this to test the setup:
```bash
python ../../test_experiments_setup.py
```

All imports should pass ✅

## 📊 What's Different from research_42d?

The `experiments` folder was the **original** location. We now primarily develop in `research_42d/experiments/`. 

This folder is kept for:
- Backwards compatibility
- Independent testing
- Comparison of older vs newer approaches

**For active development, use `research_42d/experiments/singularity_42d/`**
