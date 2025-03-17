#!/bin/bash

# Exit on error
set -e

echo "Setting up environment..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q -U pillow opencv-python-headless
pip install -q -U google-genai
echo "Python dependencies installed successfully"

# Install GOT-OCR dependencies
echo "Installing GOT-OCR dependencies..."
pip install -q -U torch==2.0.1 torchvision==0.15.2 transformers==4.37.2 tiktoken==0.6.0 verovio==4.3.1 accelerate==0.28.0 safetensors==0.4.3
echo "GOT-OCR dependencies installed successfully"

echo "Setup completed" 