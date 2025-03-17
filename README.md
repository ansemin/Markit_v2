---
title: Markit GOT OCR
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.14.0
app_file: app.py
build_script: build.sh
startup_script: setup.sh
pinned: false
---

# Markit: Document to Markdown Converter

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Ansemin101/Markit)

**Author: Anse Min** | [GitHub](https://github.com/ansemin) | [LinkedIn](https://www.linkedin.com/in/ansemin/)

## Project Links
- **GitHub Repository**: [github.com/ansemin/Markit_HF](https://github.com/ansemin/Markit_HF)
- **Hugging Face Space**: [huggingface.co/spaces/Ansemin101/Markit](https://huggingface.co/spaces/Ansemin101/Markit)

## Overview
Markit is a powerful tool that converts various document formats (PDF, DOCX, images, etc.) to Markdown format. It uses different parsing engines and OCR methods to extract text from documents and convert them to clean, readable Markdown formats.

## Key Features
- **Multiple Document Formats**: Convert PDFs, Word documents, images, and other document formats
- **Versatile Output Formats**: Export to Markdown, JSON, plain text, or document tags format
- **Advanced Parsing Engines**:
  - **PyPdfium**: Fast PDF parsing using the PDFium engine
  - **Docling**: Advanced document structure analysis
  - **Gemini Flash**: AI-powered conversion using Google's Gemini API
  - **GOT-OCR**: State-of-the-art OCR model for images (JPG/PNG only)
- **OCR Integration**: Extract text from images and scanned documents using Tesseract OCR
- **Interactive UI**: User-friendly Gradio interface with page navigation for large documents
- **AI-Powered Chat**: Interact with your documents using AI to ask questions about content

## System Architecture
The application is built with a modular architecture:
- **Core Engine**: Handles document conversion and processing workflows
- **Parser Registry**: Central registry for all document parsers
- **UI Layer**: Gradio-based web interface
- **Service Layer**: Handles AI chat functionality and external services integration

## Installation

### For Local Development
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Tesseract OCR (required for OCR functionality):
   - Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr libtesseract-dev`
   - macOS: `brew install tesseract`

4. Run the application:
   ```bash
   python app.py
   ```

### API Keys Setup

#### Gemini Flash Parser
To use the Gemini Flash parser, you need to:
1. Install the Google Generative AI client: `pip install google-genai`
2. Set the API key environment variable:
   ```bash
   # On Windows
   set GOOGLE_API_KEY=your_api_key_here
   
   # On Linux/Mac
   export GOOGLE_API_KEY=your_api_key_here
   ```
3. Alternatively, create a `.env` file in the project root with:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
4. Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

#### GOT-OCR Parser
The GOT-OCR parser requires:
1. CUDA-capable GPU with sufficient memory
2. The following dependencies will be installed automatically:
   ```bash
   torch>=2.0.1
   torchvision>=0.15.2
   transformers>=4.37.2,<4.48.0  # Specific version range required
   tiktoken>=0.6.0
   verovio>=4.3.1
   accelerate>=0.28.0
   ```
3. Note that GOT-OCR only supports JPG and PNG image formats

## Deploying to Hugging Face Spaces

### Environment Configuration
1. Go to your Space settings: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME/settings`
2. Add the following repository secrets:
   - Name: `GOOGLE_API_KEY`
   - Value: Your Gemini API key

### Space Configuration
Ensure your Hugging Face Space configuration includes:
```yaml
build:
  dockerfile: Dockerfile
  python_version: "3.10" 
  system_packages:
    - "tesseract-ocr"
    - "libtesseract-dev"
```

## How to Use

### Document Conversion
1. Upload your document using the file uploader
2. Select a parser provider:
   - **PyPdfium**: Best for standard PDFs with selectable text
   - **Docling**: Best for complex document layouts
   - **Gemini Flash**: Best for AI-powered conversions (requires API key)
   - **GOT-OCR**: Best for high-quality OCR on images (JPG/PNG only)
3. Choose an OCR option based on your selected parser:
   - **None**: No OCR processing (for documents with selectable text)
   - **Tesseract**: Basic OCR using Tesseract
   - **Advanced**: Enhanced OCR with layout preservation (available with specific parsers)
4. Select your desired output format:
   - **Markdown**: Clean, readable markdown format
   - **JSON**: Structured data representation
   - **Text**: Plain text extraction
   - **Document Tags**: XML-like structure tags
5. Click "Convert" to process your document
6. Navigate through pages using the navigation buttons for multi-page documents
7. Download the converted content in your selected format

## Troubleshooting

### OCR Issues
- Ensure Tesseract is properly installed and in your system PATH
- Check the TESSDATA_PREFIX environment variable is set correctly
- Verify language files are available in the tessdata directory

### Gemini Flash Parser Issues
- Confirm your API key is set correctly as an environment variable
- Check for API usage limits or restrictions
- Verify the document format is supported by the Gemini API

### GOT-OCR Parser Issues
- Ensure you have a CUDA-capable GPU with sufficient memory
- Verify that all required dependencies are installed correctly
- Remember that GOT-OCR only supports JPG and PNG image formats
- If you encounter CUDA out-of-memory errors, try using a smaller image
- GOT-OCR requires transformers version <4.48.0 due to API changes in newer versions
- If you see errors about 'get_max_length', downgrade transformers to version 4.47.0

### General Issues
- Check the console logs for error messages
- Ensure all dependencies are installed correctly
- For large documents, try processing fewer pages at a time

## Development Guide

### Project Structure

```
markit/
├── app.py                  # Main application entry point
├── setup.sh                # Setup script
├── build.sh                # Build script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .env                    # Environment variables
├── .gitignore              # Git ignore file
├── .gitattributes          # Git attributes file
├── src/                    # Source code
│   ├── __init__.py         # Package initialization
│   ├── main.py             # Main module
│   ├── core/               # Core functionality
│   │   ├── __init__.py     # Package initialization
│   │   ├── converter.py    # Document conversion logic
│   │   └── parser_factory.py # Parser factory
│   ├── parsers/            # Parser implementations
│   │   ├── __init__.py     # Package initialization
│   │   ├── parser_interface.py # Parser interface
│   │   ├── parser_registry.py # Parser registry
│   │   ├── docling_parser.py # Docling parser
│   │   └── pypdfium_parser.py # PyPDFium parser
│   ├── ui/                 # User interface
│   │   ├── __init__.py     # Package initialization
│   │   └── ui.py           # Gradio UI implementation
│   └── services/           # External services
│       └── __init__.py     # Package initialization
└── tests/                  # Tests
    └── __init__.py         # Package initialization
```