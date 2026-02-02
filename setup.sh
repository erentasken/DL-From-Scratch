#!/bin/bash

# Deep Learning Exercises - Setup Script
# This script sets up the development environment

echo "========================================"
echo "Deep Learning Exercises - Setup"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python 3.8+ is available
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

echo "✓ Python version OK"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest pytest-cov ipython jupyter
echo "✓ Development dependencies installed"
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p CNN/trained
mkdir -p Regularization_Recurrent/trained
mkdir -p Regularization_Recurrent/Data
echo "✓ Directories created"
echo ""

# Run tests to verify setup
echo "Running tests to verify setup..."
echo ""

echo "Testing NumPy module..."
cd numpy
python -m pytest NumpyTests.py -v --tb=short || echo "⚠ Some NumPy tests failed"
cd ..
echo ""

echo "Testing Feed-Forward Network..."
cd FeedForwardNeuralNetwork
python -m pytest NeuralNetworkTests.py -v --tb=short || echo "⚠ Some Feed-Forward tests failed"
cd ..
echo ""

echo "Testing CNN..."
cd CNN
python -m pytest NeuralNetworkTests.py -v --tb=short || echo "⚠ Some CNN tests failed"
cd ..
echo ""

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the virtual environment, run:"
echo "    source venv/bin/activate"
echo ""
echo "To run tests:"
echo "    pytest                          # Run all tests"
echo "    pytest <module>/Tests.py -v    # Run specific module tests"
echo ""
echo "To start Jupyter:"
echo "    jupyter notebook"
echo ""
echo "Happy coding! 🚀"
