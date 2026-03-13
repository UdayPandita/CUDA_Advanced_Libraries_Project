#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}GPU Edge Detection - Execution${NC}"
echo -e "${BLUE}================================${NC}"
echo

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 detected${NC}"

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}  Version: ${PYTHON_VERSION}${NC}"
echo

echo -e "${BLUE}Checking project structure...${NC}"

if [ ! -d "$SCRIPT_DIR/misc" ]; then
    echo -e "${RED}✗ 'misc' directory not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Input images directory found${NC}"

mkdir -p "$SCRIPT_DIR/output"
mkdir -p "$SCRIPT_DIR/results"
echo -e "${GREEN}✓ Output directories created${NC}"
echo

echo -e "${BLUE}Checking Python dependencies...${NC}"

REQUIRED_PACKAGES=("numpy" "cv2" "scipy" "pandas")

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✓ $package installed${NC}"
    else
        echo -e "${YELLOW}⚠ $package not found${NC}"
    fi
done

if python3 -c "import pycuda" 2>/dev/null; then
    echo -e "${GREEN}✓ pycuda installed (GPU acceleration enabled)${NC}"
else
    echo -e "${YELLOW}⚠ pycuda not installed (GPU acceleration disabled)${NC}"
fi

echo

IMAGE_COUNT=$(find "$SCRIPT_DIR/misc" -type f \( -name "*.tiff" -o -name "*.png" -o -name "*.jpg" \) | wc -l)
echo -e "${BLUE}Found ${IMAGE_COUNT} images to process${NC}"
echo

echo -e "${BLUE}Starting edge detection pipeline...${NC}"
echo -e "${BLUE}================================${NC}"
echo

cd "$SCRIPT_DIR"
python3 src/edge_detection.py

if [ $? -eq 0 ]; then
    echo
    echo -e "${BLUE}================================${NC}"
    echo -e "${GREEN}✓ Execution completed successfully!${NC}"
    echo -e "${BLUE}================================${NC}"
    echo
    
    echo -e "${BLUE}Output files:${NC}"
    echo -e "  Edge images: ${SCRIPT_DIR}/output/"
    echo -e "  Results CSV: ${SCRIPT_DIR}/results/execution_times.csv"
    echo
    
    if [ -f "$SCRIPT_DIR/results/execution_times.csv" ]; then
        echo -e "${BLUE}Results Preview:${NC}"
        head -10 "$SCRIPT_DIR/results/execution_times.csv" | sed 's/^/  /'
        echo
    fi
else
    echo
    echo -e "${RED}✗ Execution failed${NC}"
    exit 1
fi

echo -e "${GREEN}Done!${NC}"
