#!/bin/bash

# AIDIMIN Setup Script
# This script sets up the AIDIMIN prediction tool on Ubuntu

echo "=========================================="
echo "AIDIMIN Setup Script"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    echo "   sudo apt update && sudo apt install python3 python3-pip"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
required_version="3.7"

if [[ $(echo "$python_version >= $required_version" | bc -l) -eq 0 ]]; then
    echo "❌ Python version $python_version is too old. Requires Python $required_version+"
    exit 1
fi

echo "✓ Python $python_version detected"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Installing..."
    sudo apt update
    sudo apt install python3-pip -y
fi

echo "✓ pip3 is available"

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (recommended) [y/N]: " create_venv

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv aidimin_env
    
    echo "Activating virtual environment..."
    source aidimin_env/bin/activate
    
    echo "✓ Virtual environment created and activated"
    echo "💡 To activate in future sessions: source aidimin_env/bin/activate"
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Make prediction script executable
chmod +x predict.py
echo "✓ Made predict.py executable"

# Check if model file exists
if [ ! -f "aidimin.h5" ]; then
    echo "⚠️  Model file 'aidimin.h5' not found in current directory"
    echo "   Please ensure you have the trained model file before running predictions"
else
    echo "✓ Model file found"
fi

# Create sample directory structure
mkdir -p predictions
mkdir -p examples

echo "✓ Created directory structure"

# Test installation
echo "Testing installation..."
python3 predict.py --help > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Installation successful!"
else
    echo "❌ Installation test failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Usage examples:"
echo "  python3 predict.py -f sequences.fasta"
echo "  python3 predict.py --auto"
echo "  python3 predict.py --help"
echo ""
echo "Next steps:"
echo "1. Place your .h5 model file in this directory"
echo "2. Add your FASTA files to predict"
echo "3. Run predictions!"
echo ""

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Remember to activate virtual environment:"
    echo "  source aidimin_env/bin/activate"
    echo ""
fi