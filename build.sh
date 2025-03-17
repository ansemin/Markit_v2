#!/bin/bash

# Exit on error
set -e

echo "Starting build process..."

# Install system dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    wget \
    pkg-config

# Install Google Gemini API client
echo "Installing Google Gemini API client..."
pip install -q -U google-genai
echo "Google Gemini API client installed successfully"

# Install GOT-OCR dependencies
echo "Installing GOT-OCR dependencies..."
pip install -q -U torch==2.0.1 torchvision==0.15.2 transformers==4.37.2 tiktoken==0.6.0 verovio==4.3.1 accelerate==0.28.0 safetensors==0.4.3
echo "GOT-OCR dependencies installed successfully"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -e .

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env || echo "Warning: .env.example not found"
fi

echo "Build process completed successfully!"