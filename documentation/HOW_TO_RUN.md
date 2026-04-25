# 🚀 HOW TO RUN THE COSMIC DAVIS
## A Step-by-Step Guide for New Users

Welcome to the **Cosmic Davis** project. This guide will help you set up and run the 12D Hebbian Transformer on your own machine.

---

### 1️⃣ PREREQUISITES

Before you begin, ensure you have the following installed:

1.  **Python 3.8 or higher**: [Download Python](https://www.python.org/downloads/)
    *   *Make sure to check "Add Python to PATH" during installation.*
2.  **Git**: [Download Git](https://git-scm.com/downloads) (Optional, for cloning)

---

### 2️⃣ INSTALLATION

1.  **Open a Terminal** (Command Prompt or PowerShell) in the project folder.
2.  **Install Dependencies**:
    Run the following command to install all necessary libraries:
    ```bash
    pip install torch numpy scipy librosa faiss-cpu Pillow PyMuPDF matplotlib rich
    ```
    *Note: If you have an NVIDIA GPU, install the CUDA version of PyTorch for faster training.*

---

### 3️⃣ RUNNING THE SYSTEM (The Easy Way)

We have provided a **Master Launcher** for Windows users.

1.  Double-click **`START_COSMIC_DAVIS.bat`**.
2.  A menu will appear with all options:

**Training Options:**
   *   **[1] Standard Training:** Single-mind autonomous learning
   *   **[2] Dual Mind Training:** 12D + 42D debate & evolution system

**Monitoring:**
   *   **[3] Neural Monitor:** Real-time brain visualization
   *   **[4] Dual Mind Showcase:** Interactive demonstration

**Chat Options:**
   *   **[5] Dual Mind Chat:** Talk to both 12D + 42D together
   *   **[6] 12D Only:** Chat with grounded model
   *   **[7] 42D Only:** Chat with abstract model

**Recommended First-Time Workflow:**
1.  Select **[4] Showcase** to see the Dual Mind system demo
2.  Select **[2] Dual Mind Training** and let it run in background
3.  Select **[3] Monitor** to watch it learn
4.  Select **[5] Dual Mind Chat** to interact with both minds

---

### 4️⃣ TROUBLESHOOTING

**"ModuleNotFoundError: No module named 'cosmic_synapse'"**
*   **Cause:** Python cannot find the core library.
*   **Fix:** Use the provided `.bat` files. They automatically set the `PYTHONPATH` for you.

**"Vocabulary not found"**
*   **Cause:** You haven't started training yet.
*   **Fix:** Run `start_training.bat` and wait for it to say "FULL VOCABULARY INITIALIZED".

**"Checkpoint not found"**
*   **Cause:** The model hasn't saved its first brain state yet.
*   **Fix:** Let `start_training.bat` run for at least 1-2 minutes until it says "Checkpoints saved".

**"CUDA Available: False"**
*   **Cause:** You are running on CPU (slower but works).
*   **Fix:** If you have a GPU, install PyTorch with CUDA support manually.

---

### 5️⃣ FOR DEVELOPERS (Manual Command Line)

If you prefer using the terminal manually, you **MUST** set the `PYTHONPATH` before running scripts.

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "$PWD\packages\cosmic-synapse-transformer"; python research_42d/experiments/singularity_42d/autonomous_study.py
```

**Windows (CMD):**
```cmd
set PYTHONPATH=%CD%\packages\cosmic-synapse-transformer
python research_42d/experiments/singularity_42d/autonomous_study.py
```

**Linux/Mac:**
```bash
export PYTHONPATH=$PWD/packages/cosmic-synapse-transformer
python research_42d/experiments/singularity_42d/autonomous_study.py
```
