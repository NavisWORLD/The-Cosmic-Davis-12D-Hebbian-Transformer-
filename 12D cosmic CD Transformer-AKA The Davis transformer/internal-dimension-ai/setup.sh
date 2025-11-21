#!/bin/bash
# Internal Dimension AI - Easy Setup Script
# For researchers without programming experience
# This script handles all installation automatically

set -e  # Exit on error

echo "========================================================================="
echo "  INTERNAL DIMENSION AI - AUTOMATIC SETUP"
echo "========================================================================="
echo ""
echo "This script will:"
echo "  1. Check your Python version"
echo "  2. Create a virtual environment (isolated Python workspace)"
echo "  3. Install all required packages"
echo "  4. Verify everything is working"
echo ""
echo "This may take 5-10 minutes depending on your internet speed."
echo ""
read -p "Press ENTER to start setup, or Ctrl+C to cancel..."
echo ""

# Step 1: Check Python version
echo "Step 1/4: Checking Python version..."
echo "-----------------------------------------------------------------------"

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3.8 or higher from: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION ✓"
echo ""

# Step 2: Create virtual environment
echo "Step 2/4: Creating virtual environment..."
echo "-----------------------------------------------------------------------"

if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "Virtual environment created ✓"
fi
echo ""

# Step 3: Activate and install packages
echo "Step 3/4: Installing packages (this may take several minutes)..."
echo "-----------------------------------------------------------------------"

source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip --quiet

# Install packages
echo "Installing required packages..."
pip install -r requirements.txt --quiet

echo "All packages installed ✓"
echo ""

# Step 4: Verify installation
echo "Step 4/4: Verifying installation..."
echo "-----------------------------------------------------------------------"

python3 << 'EOF'
import sys
import pkg_resources

required = {
    'torch': '2.0.0',
    'numpy': '1.24.0',
    'pandas': '2.0.0',
    'matplotlib': '3.7.0',
    'seaborn': '0.12.0',
    'tqdm': '4.65.0',
    'scikit-learn': '1.2.0',
}

print("Checking installed packages:")
all_good = True
for package, min_version in required.items():
    try:
        installed = pkg_resources.get_distribution(package).version
        print(f"  {package:20s} {installed:15s} ✓")
    except pkg_resources.DistributionNotFound:
        print(f"  {package:20s} NOT FOUND ✗")
        all_good = False

if all_good:
    print("\nAll packages verified ✓")
else:
    print("\nSome packages are missing. Please run setup again.")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================="
    echo "  SETUP COMPLETE! ✓"
    echo "========================================================================="
    echo ""
    echo "Everything is ready to use!"
    echo ""
    echo "To get started:"
    echo "  1. Run: source venv/bin/activate"
    echo "  2. Run: python run_easy.py"
    echo ""
    echo "Or use the quick start script:"
    echo "  ./start.sh"
    echo ""
else
    echo ""
    echo "Setup encountered errors. Please check the output above."
    exit 1
fi
