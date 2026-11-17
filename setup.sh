#!/bin/bash
# ============================================================
#  DRS Project Setup Script
#  Run this after Xcode Command Line Tools are installed.
# ============================================================

set -e

echo "============================================"
echo "  DRS Project - Setup"
echo "============================================"
echo

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found."
    echo "Please install Xcode Command Line Tools first:"
    echo "  xcode-select --install"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"

# Create virtual environment
echo
echo "[1/3] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  Virtual environment already exists. Skipping."
else
    python3 -m venv venv
    echo "  Created: ./venv"
fi

# Activate virtual environment
echo "[2/3] Activating virtual environment..."
source venv/bin/activate
echo "  Active: $(which python)"

# Install dependencies
echo "[3/3] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo
echo "To run the DRS system:"
echo "  python main.py"
echo
echo "To run with options:"
echo "  python main.py --input video.mp4 --show-preview"
echo
