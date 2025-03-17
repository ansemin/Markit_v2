#!/bin/bash

# Exit on error
set -e

echo "Starting build process..."

# Install system dependencies for tesseract
echo "Installing Tesseract and dependencies..."
apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    wget

# Create tessdata directory
TESSDATA_DIR="/usr/share/tesseract-ocr/4.00/tessdata"
mkdir -p "$TESSDATA_DIR"

# Download traineddata files directly from the official repository
echo "Downloading Tesseract traineddata files..."
wget -O "$TESSDATA_DIR/eng.traineddata" "https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata"
wget -O "$TESSDATA_DIR/osd.traineddata" "https://github.com/tesseract-ocr/tessdata/raw/main/osd.traineddata"

# Set and verify TESSDATA_PREFIX
export TESSDATA_PREFIX="$TESSDATA_DIR"
echo "TESSDATA_PREFIX=${TESSDATA_PREFIX}" >> /etc/environment

# Verify tesseract installation and data files
echo "Verifying Tesseract installation..."
if ! command -v tesseract &> /dev/null; then
    echo "Tesseract installation failed!"
    exit 1
fi
echo "Tesseract version: $(tesseract --version)"

# Verify traineddata files
echo "Verifying traineddata files..."
if [ ! -f "$TESSDATA_DIR/eng.traineddata" ]; then
    echo "eng.traineddata is missing!"
    exit 1
fi
if [ ! -f "$TESSDATA_DIR/osd.traineddata" ]; then
    echo "osd.traineddata is missing!"
    exit 1
fi

echo "Traineddata files in $TESSDATA_DIR:"
ls -l "$TESSDATA_DIR"

# Test Tesseract functionality
echo "Testing Tesseract functionality..."
echo "Hello World" > test.png
if ! tesseract test.png stdout; then
    echo "Tesseract test failed!"
    exit 1
fi
rm test.png

# Clean and install tesserocr from source
echo "Installing tesserocr from source..."
pip uninstall -y tesserocr || true
CPPFLAGS="-I/usr/include/tesseract" LDFLAGS="-L/usr/lib/x86_64-linux-gnu/" pip install --no-binary :all: tesserocr

# Verify tesserocr installation
echo "Verifying tesserocr installation..."
python3 -c "
import tesserocr
print(f'tesserocr version: {tesserocr.__version__}')
print(f'Available languages: {tesserocr.get_languages()}')
print(f'TESSDATA_PREFIX: {tesserocr.get_languages()[1]}')
"

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