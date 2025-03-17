#!/bin/bash

# Exit on error
set -e

echo "Setting up Tesseract OCR environment..."

# Install required packages if not already installed
if ! command -v tesseract &> /dev/null; then
    echo "Tesseract not found, attempting to install..."
    apt-get update -y || echo "Failed to update apt, continuing anyway"
    apt-get install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev || echo "Failed to install tesseract via apt, continuing anyway"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q -U pytesseract pillow opencv-python-headless pdf2image
pip install -q -U google-genai
echo "Python dependencies installed successfully"

# Install GOT-OCR dependencies
echo "Installing GOT-OCR dependencies..."
pip install -q -U torch==2.0.1 torchvision==0.15.2 transformers==4.37.2 tiktoken==0.6.0 verovio==4.3.1 accelerate==0.28.0 safetensors==0.4.1
echo "GOT-OCR dependencies installed successfully"

# Install tesserocr with pip
echo "Installing tesserocr..."
pip install -q -U tesserocr || echo "Failed to install tesserocr with pip, trying with specific compiler flags..."

# If tesserocr installation failed, try with specific compiler flags
if ! python -c "import tesserocr" &> /dev/null; then
    echo "Trying alternative tesserocr installation..."
    CPPFLAGS="-I/usr/local/include -I/usr/include" LDFLAGS="-L/usr/local/lib -L/usr/lib" pip install -q -U tesserocr || echo "Failed to install tesserocr with compiler flags, continuing anyway"
fi

# Create tessdata directory if it doesn't exist
mkdir -p tessdata

# Set TESSDATA_PREFIX environment variable
export TESSDATA_PREFIX="$(pwd)/tessdata"
echo "TESSDATA_PREFIX set to: $TESSDATA_PREFIX"

# Download eng.traineddata if it doesn't exist
if [ ! -f "tessdata/eng.traineddata" ]; then
  echo "Downloading eng.traineddata..."
  wget -O tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata || \
  curl -o tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
  echo "Downloaded eng.traineddata"
else
  echo "eng.traineddata already exists"
fi

# Try to copy to system locations (may fail in restricted environments)
for tessdata_dir in "/usr/share/tesseract-ocr/4.00/tessdata" "/usr/share/tesseract-ocr/tessdata" "/usr/local/share/tessdata"; do
  if [ -d "$tessdata_dir" ]; then
    echo "Copying eng.traineddata to $tessdata_dir..."
    cp -f tessdata/eng.traineddata "$tessdata_dir/" 2>/dev/null || echo "Failed to copy to $tessdata_dir, continuing anyway"
  fi
done

# Verify Tesseract installation
echo "Verifying Tesseract installation..."
tesseract --version || echo "Tesseract not found in PATH, but may still be available to Python"

# Test tesserocr if installed
echo "Testing tesserocr..."
python -c "import tesserocr; print(f'tesserocr version: {tesserocr.tesseract_version()}')" || echo "tesserocr not working, but may still be able to use pytesseract"

# Test pytesseract
echo "Testing pytesseract..."
python -c "import pytesseract; print(f'pytesseract path: {pytesseract.tesseract_cmd}')" || echo "pytesseract not working"

echo "Setup completed"

# Add TESSDATA_PREFIX to .env file for persistence
echo "TESSDATA_PREFIX=$(pwd)/tessdata" >> .env 