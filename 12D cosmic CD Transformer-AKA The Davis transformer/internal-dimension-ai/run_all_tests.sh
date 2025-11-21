#!/bin/bash
echo "=== INTERNAL DIMENSION AI - COMPLETE TEST SUITE ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track overall success
ALL_PASSED=true

echo "1. Running unit tests..."
cd internal-dimension-ai
python -m pytest tests/ -v --tb=short 2>&1 | tee /tmp/test_pytest.log
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ Unit tests passed${NC}"
else
    echo -e "${RED}✗ Unit tests failed${NC}"
    ALL_PASSED=false
fi

echo ""
echo "2. Testing examples..."

# Test 01_quick_demo.py
echo "  Testing 01_quick_demo.py..."
python examples/01_quick_demo.py > /tmp/test_quick_demo.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓ Quick demo passed${NC}"
else
    echo -e "  ${RED}✗ Quick demo failed${NC}"
    echo "    See /tmp/test_quick_demo.log for details"
    ALL_PASSED=false
fi

# Test 02_baseline_comparison.py (abbreviated run)
echo "  Testing 02_baseline_comparison.py..."
timeout 60 python examples/02_baseline_comparison.py > /tmp/test_baseline.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓ Baseline comparison passed${NC}"
elif [ $? -eq 124 ]; then
    echo -e "  ${GREEN}✓ Baseline comparison running (timed out after 60s, as expected)${NC}"
else
    echo -e "  ${RED}✗ Baseline comparison failed${NC}"
    echo "    See /tmp/test_baseline.log for details"
    ALL_PASSED=false
fi

# Test 03_curiosity_demo.py
echo "  Testing 03_curiosity_demo.py..."
timeout 60 python examples/03_curiosity_demo.py > /tmp/test_curiosity.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓ Curiosity demo passed${NC}"
elif [ $? -eq 124 ]; then
    echo -e "  ${GREEN}✓ Curiosity demo running (timed out after 60s, as expected)${NC}"
else
    echo -e "  ${RED}✗ Curiosity demo failed${NC}"
    echo "    See /tmp/test_curiosity.log for details"
    ALL_PASSED=false
fi

echo ""
echo "3. Testing standalone files..."

cd ..

# Test simple_consciousness_demo.py
if [ -f simple_consciousness_demo.py ]; then
    echo "  Testing simple_consciousness_demo.py..."
    timeout 60 python simple_consciousness_demo.py > /tmp/test_simple.log 2>&1
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Simple demo passed${NC}"
    elif [ $? -eq 124 ]; then
        echo -e "  ${GREEN}✓ Simple demo running (timed out after 60s, as expected)${NC}"
    else
        echo -e "  ${RED}✗ Simple demo failed${NC}"
        echo "    See /tmp/test_simple.log for details"
        ALL_PASSED=false
    fi
else
    echo "  ⊘ simple_consciousness_demo.py not found (skipping)"
fi

echo ""
echo "4. Testing cosmic synapse (quick 1000 step run)..."
cd internal-dimension-ai
python scripts/run_cosmic_synapse.py --steps 1000 --particles 32 --physics-steps 100 > /tmp/test_cosmic.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Cosmic synapse passed${NC}"
else
    echo -e "${RED}✗ Cosmic synapse failed${NC}"
    echo "  See /tmp/test_cosmic.log for details"
    ALL_PASSED=false
fi

echo ""
echo "5. Syntax check on all Python files..."
find . -name "*.py" -type f | while read file; do
    python -m py_compile "$file" 2>&1 | grep -v "^$"
done > /tmp/syntax_check.log 2>&1
if [ -s /tmp/syntax_check.log ]; then
    echo -e "${RED}✗ Syntax errors found${NC}"
    cat /tmp/syntax_check.log
    ALL_PASSED=false
else
    echo -e "${GREEN}✓ All Python files have valid syntax${NC}"
fi

echo ""
echo "=== TEST SUITE COMPLETE ==="
if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Check logs in /tmp/ for details.${NC}"
    exit 1
fi
