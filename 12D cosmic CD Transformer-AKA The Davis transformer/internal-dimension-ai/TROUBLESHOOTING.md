# Troubleshooting Guide - Internal Dimension AI

This guide helps you solve common problems you might encounter.

---

## 🔧 Setup Problems

### Problem: "Permission denied" when running setup.sh

**Solution**: Make the script executable first
```bash
chmod +x setup.sh
chmod +x start.sh
./setup.sh
```

### Problem: "Python 3 is not installed" or "command not found"

**Solution**: Install Python 3.8 or higher

**On macOS:**
```bash
brew install python3
```

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv
```

**On Windows:**
- Download from https://www.python.org/downloads/
- Make sure to check "Add Python to PATH" during installation

### Problem: "pip: command not found"

**Solution**: Install pip
```bash
python3 -m ensurepip --upgrade
```

Or on Ubuntu/Debian:
```bash
sudo apt-get install python3-pip
```

### Problem: Setup fails with "No module named 'venv'"

**Solution**: Install python3-venv
```bash
# Ubuntu/Debian
sudo apt-get install python3-venv

# macOS (usually included, but if needed)
python3 -m pip install virtualenv
```

### Problem: Installation is very slow

**This is normal!** The packages (especially PyTorch) are large. On a slow internet connection, installation can take 10-30 minutes. Be patient and let it finish.

---

## 🚀 Running Problems

### Problem: "No module named 'torch'" or "No module named 'numpy'"

**Solution**: Activate the virtual environment first
```bash
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

Then run your command. Or just use:
```bash
./start.sh    # This activates automatically
```

### Problem: Script runs but crashes immediately

**Solution**: Check if you activated the environment
```bash
# Make sure you see (venv) at the start of your command prompt
source venv/bin/activate
python run_easy.py
```

### Problem: "CUDA out of memory" error

**Solution**: Your GPU doesn't have enough memory. Use CPU instead.

Edit the config file or use CPU mode:
```python
# The system will automatically use CPU if CUDA is not available
# No action needed - it will just be slower
```

Or reduce batch size in the configuration files.

### Problem: Training is very slow

**This might be normal!** Reinforcement learning takes time:
- Quick demo: 5 minutes
- Baseline comparison: 15 minutes
- Full experiments: hours

**To speed up**:
1. Use a GPU if available (much faster)
2. Reduce number of episodes in config files
3. Use fewer replications for testing

---

## 📊 Results Problems

### Problem: "No results found" when trying to visualize

**Solution**: You need to run an experiment first!
```bash
./start.sh
# Select option 1 (Quick Demo) first
# Then you can visualize results
```

### Problem: Plots don't show up

**Solution**: Check if the figures were saved to disk
```bash
ls outputs/*/
# You should see .png files

# Open them directly:
open outputs/quick_demo/quick_demo_results.png  # macOS
xdg-open outputs/quick_demo/quick_demo_results.png  # Linux
```

If using SSH/remote server, copy files to your local machine:
```bash
scp -r user@server:/path/to/internal-dimension-ai/outputs ./
```

### Problem: Results look wrong or metrics are strange

**Expected ranges**:
- Rewards: Depends on environment (usually -10 to 10)
- R_ω (Synaptic Diversity): -1 to 1 (optimal: 0.5-0.7)
- R_ψ (Phase Coherence): 0 to 1 (higher is better)
- φ (Integration): 0 to 1 (higher is better)
- Consciousness Score: 0 to 1 (higher indicates more complexity)

If values are NaN or infinite, there might be a numerical instability. Try:
- Using smaller learning rate
- Reducing internal dimension size
- Checking for bugs in custom code

---

## 💻 System-Specific Issues

### macOS: "command not found: ./setup.sh"

**Solution**: Make sure you're in the right directory
```bash
cd /path/to/internal-dimension-ai
pwd  # Should show .../internal-dimension-ai
./setup.sh
```

### Windows: Scripts don't work

**Solution**: Use Python directly
```bash
# Instead of ./setup.sh
python setup_windows.py  # If available

# Or install manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Then run:
python run_easy.py
```

### Linux: "Permission denied" for everything

**Solution**: Don't use sudo. Run as your user:
```bash
# Make scripts executable
chmod +x setup.sh start.sh

# Run as yourself (not root)
./setup.sh
```

---

## 🔍 Debugging Tips

### Enable Verbose Logging

Edit `run_easy.py` or any example script to add:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Dependencies

Run the requirements check:
```bash
source venv/bin/activate
python run_easy.py
# Select option 9 (Check System Requirements)
```

### Test Individual Components

```bash
source venv/bin/activate

# Test imports
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "from src.core.network import InternalDimensionNetwork; print('IDN: OK')"

# Test environment
python -c "from src.environments.gridworld import GridWorld; env = GridWorld(); print('Environment: OK')"
```

### Get More Help

If you're still stuck:

1. **Check the log files** in `logs/` directory
2. **Read error messages carefully** - they usually tell you what's wrong
3. **Make sure you're using Python 3.8+**
4. **Ensure virtual environment is activated** (you should see `(venv)` in prompt)
5. **Try reinstalling**:
   ```bash
   rm -rf venv
   ./setup.sh
   ```

---

## 📝 Common Error Messages

### "RuntimeError: CUDA error: out of memory"
→ Your GPU is full. The system will automatically use CPU. Just restart.

### "ImportError: cannot import name 'InternalDimensionNetwork'"
→ Virtual environment not activated. Run `source venv/bin/activate`

### "FileNotFoundError: [Errno 2] No such file or directory"
→ Wrong directory. Navigate to `internal-dimension-ai` folder first.

### "numpy.core._exceptions._ArrayMemoryError: Unable to allocate array"
→ Your system is out of RAM. Close other programs or reduce batch size.

### "ModuleNotFoundError: No module named 'torch'"
→ Dependencies not installed or environment not activated.

---

## ✅ Quick Checklist

Before asking for help, verify:

- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] In correct directory (`pwd` shows .../internal-dimension-ai)
- [ ] Setup script completed successfully
- [ ] Virtual environment activated (`(venv)` in prompt)
- [ ] No typos in commands
- [ ] Read the error message carefully

---

## 🆘 Still Need Help?

If none of this works:

1. **Document your problem**:
   - What command did you run?
   - What was the exact error message?
   - What operating system?
   - What Python version?

2. **Check existing issues** in RESEARCH_PLAN.md

3. **Start fresh**:
   ```bash
   rm -rf venv __pycache__ outputs checkpoints
   ./setup.sh
   ```

Good luck! 🚀
