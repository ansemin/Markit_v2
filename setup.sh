#!/bin/bash

# Exit on error
set -e

echo "Starting setup process..."

# Enable more verbose logging
set -x

# Check if running with sudo/root permissions for system dependencies
if [ "$EUID" -eq 0 ]; then
    # Install system dependencies
    echo "Installing system dependencies..."
    apt-get update && apt-get install -y \
        wget \
        pkg-config \
        git \
        tree  # Add tree for directory structure visualization
    echo "System dependencies installed successfully"
else
    echo "Not running as root. Skipping system dependencies installation."
    echo "Make sure git is installed on your system for GOT-OCR to work properly."
fi

# Install NumPy first as it's required by many other packages
echo "Installing NumPy..."
pip install -q -U "numpy<2.0.0" --no-cache-dir
echo "NumPy installed successfully"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q -U pillow opencv-python-headless
pip install -q -U google-genai
pip install -q -U latex2markdown
echo "Python dependencies installed successfully"

# Install GOT-OCR dependencies
echo "Installing GOT-OCR dependencies..."
pip install -q -U torch==2.0.1 torchvision==0.15.2 transformers==4.37.2 tiktoken==0.6.0 verovio==4.3.1 accelerate==0.28.0 safetensors==0.4.3 huggingface_hub
echo "GOT-OCR dependencies installed successfully"

# Install Hugging Face CLI
echo "Installing Hugging Face CLI..."
pip install -q -U "huggingface_hub[cli]"
echo "Hugging Face CLI installed successfully"

# Add debug section for GOT-OCR repo
echo "===== GOT-OCR Repository Debugging ====="

# Clone the repository for inspection (if it doesn't exist)
TEMP_DIR="/tmp"
REPO_DIR="${TEMP_DIR}/GOT-OCR2.0"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning GOT-OCR2.0 repository for debugging..."
    git clone https://github.com/Ucas-HaoranWei/GOT-OCR2.0.git "$REPO_DIR"
else
    echo "GOT-OCR2.0 repository already exists at $REPO_DIR"
fi

# Check the repository structure
echo "GOT-OCR2.0 repository structure:"
if command -v tree &> /dev/null; then
    tree -L 3 "$REPO_DIR"
else
    find "$REPO_DIR" -type d -maxdepth 3 | sort
fi

# Check if the demo script exists
DEMO_SCRIPT="${REPO_DIR}/GOT/demo/run_ocr_2.0.py"
if [ -f "$DEMO_SCRIPT" ]; then
    echo "Demo script found at: $DEMO_SCRIPT"
else
    echo "ERROR: Demo script not found at: $DEMO_SCRIPT"
    
    # Search for the script in the repository
    echo "Searching for run_ocr_2.0.py in the repository..."
    find "$REPO_DIR" -name "run_ocr_2.0.py" -type f
fi

echo "===== End of GOT-OCR Debugging ====="

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

# Return to normal logging
set +x

echo "Setup process completed successfully!" 