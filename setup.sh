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
        ffmpeg
    echo "System dependencies installed successfully"
else
    echo "Not running as root. Skipping system dependencies installation."
fi

# Install NumPy first as it's required by many other packages
echo "Installing NumPy..."
pip install -q -U "numpy<2.0.0" --no-cache-dir
echo "NumPy installed successfully"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q -U pillow opencv-python
pip install -q -U google-genai
pip install -q -U openai>=1.1.0  # For LLM image description support
# pip install -q -U latex2markdown - removed, now using Gemini API for LaTeX conversion
echo "Python dependencies installed successfully"

# Install GOT-OCR transformers dependencies
echo "Installing GOT-OCR transformers dependencies..."
pip install -q -U torch torchvision
pip install -q -U "git+https://github.com/huggingface/transformers.git@main" accelerate verovio
pip install -q -U "huggingface_hub[cli]>=0.19.0"
pip install -q -U "numpy==1.26.3"  # Exact version as in original
echo "GOT-OCR transformers dependencies installed successfully"

# Install spaces module for ZeroGPU support
echo "Installing spaces module for ZeroGPU support..."
pip install -q -U spaces
echo "Spaces module installed successfully"

# Install markitdown with all optional dependencies
echo "Installing MarkItDown with all dependencies..."
pip install -q -U 'markitdown[all]'
echo "MarkItDown installed successfully"

# Install Docling for advanced PDF understanding
echo "Installing Docling..."
pip install -q -U docling
echo "Docling installed successfully"

# Install LangChain and RAG dependencies
echo "Installing LangChain and RAG dependencies..."
pip install -q -U langchain>=0.3.0
pip install -q -U langchain-openai>=0.2.0
pip install -q -U langchain-google-genai>=2.0.0
pip install -q -U langchain-chroma>=0.1.0
pip install -q -U langchain-text-splitters>=0.3.0
pip install -q -U langchain-community>=0.3.0  # For BM25Retriever and EnsembleRetriever
pip install -q -U chromadb>=0.5.0
pip install -q -U sentence-transformers>=3.0.0
pip install -q -U rank-bm25>=0.2.0  # Required for BM25Retriever
echo "LangChain and RAG dependencies installed successfully"

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