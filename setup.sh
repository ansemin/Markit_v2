#!/bin/bash

# Exit on error
set -e

echo "Starting setup process..."

# Check if running with sudo/root permissions for system dependencies
if [ "$EUID" -eq 0 ]; then
    # Install system dependencies
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y \
        wget \
        pkg-config
    echo "System dependencies installed successfully"
else
    echo "Not running as root. Skipping system dependencies installation."
    echo "If system dependencies are needed, please run this script with sudo."
fi

# Install NumPy first as it's required by many other packages
echo "Installing NumPy..."
pip install -q -U "numpy<2.0.0" --no-cache-dir
echo "NumPy installed successfully"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q -U pillow opencv-python-headless
pip install -q -U google-genai
echo "Python dependencies installed successfully"

# Install GOT-OCR dependencies
echo "Installing GOT-OCR dependencies..."
pip install -q -U torch==2.0.1 torchvision==0.15.2 transformers==4.37.2 tiktoken==0.6.0 verovio==4.3.1 accelerate==0.28.0 safetensors==0.4.3
echo "GOT-OCR dependencies installed successfully"

# Install the project in development mode only if setup.py or pyproject.toml exists
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Installing project in development mode..."
    pip install -e .
    echo "Project installed successfully"
else
    echo "No setup.py or pyproject.toml found, skipping project installation"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo ".env file created from .env.example"
    else
        echo "Warning: .env.example not found. Creating empty .env file."
        touch .env
    fi
fi

echo "Setup process completed successfully!" 