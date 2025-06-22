---
title: Markit_v2
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.14.0
app_file: app.py
build_script: build.sh
startup_script: setup.sh
pinned: false
hf_oauth: true
---

# Document to Markdown Converter

A Hugging Face Space that converts various document formats to Markdown, now with MarkItDown integration!

## Features

- Convert PDFs, Office documents, images, and more to Markdown
- Multiple parser options:
  - MarkItDown: For comprehensive document conversion
  - GOT-OCR: For image-based OCR with LaTeX support
  - Gemini Flash: For AI-powered text extraction
- Download converted documents as Markdown files
- Clean, responsive UI

## Using MarkItDown

This app integrates [Microsoft's MarkItDown](https://github.com/microsoft/markitdown) library, which supports a wide range of file formats:

- PDF
- PowerPoint (PPTX)
- Word (DOCX)
- Excel (XLSX)
- Images (JPG, PNG)
- Audio files (with transcription)
- HTML
- Text-based formats (CSV, JSON, XML)
- ZIP files
- YouTube URLs
- EPubs
- And more!

## Environment Variables

The application uses centralized configuration management. You can enhance functionality by setting these environment variables:

### 🔑 **API Keys:**
- `GOOGLE_API_KEY`: Used for Gemini Flash parser and LaTeX to Markdown conversion
- `OPENAI_API_KEY`: Enables AI-based image descriptions in MarkItDown
- `MISTRAL_API_KEY`: For Mistral OCR parser (if available)

### ⚙️ **Configuration Options:**
- `DEBUG`: Set to `true` for debug mode with verbose logging
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: 10MB)
- `TEMP_DIR`: Directory for temporary files (default: ./temp)
- `TESSERACT_PATH`: Custom path to Tesseract executable
- `TESSDATA_PATH`: Path to Tesseract language data

### 🤖 **Model Configuration:**
- `GEMINI_MODEL`: Gemini model to use (default: gemini-1.5-flash)
- `MISTRAL_MODEL`: Mistral model to use (default: pixtral-12b-2409)
- `GOT_OCR_MODEL`: GOT-OCR model to use (default: stepfun-ai/GOT-OCR2_0)
- `MODEL_TEMPERATURE`: Model temperature for AI responses (default: 0.1)
- `MODEL_MAX_TOKENS`: Maximum tokens for AI responses (default: 4096)

## Usage

1. Select a file to upload
2. Choose "MarkItDown" as the parser
3. Select "Standard Conversion"
4. Click "Convert"
5. View the Markdown output and download the converted file

## Local Development

### 🚀 **Quick Start:**
1. Clone the repository
2. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   MISTRAL_API_KEY=your_mistral_api_key_here
   DEBUG=true
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   # For full environment setup (HF Spaces compatible)
   python app.py
   
   # For local development (faster startup)
   python run_app.py
   ```

### 🧪 **Development Features:**
- **Automatic Environment Setup**: Dependencies are checked and installed automatically
- **Configuration Validation**: Startup validation reports missing API keys and configuration issues
- **Enhanced Error Messages**: Detailed error reporting for debugging
- **Centralized Logging**: Configurable logging levels and output formats

## Credits

- [MarkItDown](https://github.com/microsoft/markitdown) by Microsoft
- [GOT-OCR](https://github.com/stepfun-ai/GOT-OCR-2.0) for image-based OCR
- [Gradio](https://gradio.app/) for the UI framework

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
  - **MarkItDown**: Comprehensive document conversion (PDFs, Office docs, images, audio, etc.)
  - **Gemini Flash**: AI-powered conversion using Google's Gemini API
  - **GOT-OCR**: State-of-the-art OCR model for images (JPG/PNG only) with plain text and formatted text options
  - **Mistral OCR**: Advanced OCR using Mistral's Pixtral model for image-to-text conversion
- **OCR Integration**: Extract text from images and scanned documents using Tesseract OCR
- **Interactive UI**: User-friendly Gradio interface with page navigation for large documents
- **AI-Powered Chat**: Interact with your documents using AI to ask questions about content
- **ZeroGPU Support**: Optimized for Hugging Face Spaces with Stateless GPU environments

## System Architecture

The application is built with a clean, layered architecture following modern software engineering principles:

### 🏗️ **Core Architecture Components:**
- **Entry Point** (`app.py`): HF Spaces-compatible application launcher with environment setup
- **Configuration Layer** (`src/core/config.py`): Centralized configuration management with validation
- **Service Layer** (`src/services/`): Business logic for document processing and external services
- **Core Engine** (`src/core/`): Document conversion workflows and utilities
- **Parser Registry** (`src/parsers/`): Extensible parser system with standardized interfaces
- **UI Layer** (`src/ui/`): Gradio-based web interface with enhanced error handling

### 🎯 **Key Architectural Features:**
- **Separation of Concerns**: Clean boundaries between UI, business logic, and core utilities
- **Centralized Configuration**: All settings, API keys, and validation in one place
- **Custom Exception Hierarchy**: Proper error handling with user-friendly messages
- **Plugin Architecture**: Easy addition of new document parsers
- **HF Spaces Optimized**: Maintains compatibility with Hugging Face deployment requirements

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
   torch
   torchvision
   git+https://github.com/huggingface/transformers.git@main  # Latest transformers from GitHub
   accelerate
   verovio
   numpy==1.26.3  # Specific version required
   opencv-python
   ```
3. Note that GOT-OCR only supports JPG and PNG image formats
4. In HF Spaces, the integration with ZeroGPU is automatic and optimized for Stateless GPU environments

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
   - **MarkItDown**: Best for comprehensive document conversion (supports PDFs, Office docs, images, audio, etc.)
   - **Gemini Flash**: Best for AI-powered conversions (requires API key)
   - **GOT-OCR**: Best for high-quality OCR on images (JPG/PNG only)
   - **Mistral OCR**: Advanced OCR using Mistral's Pixtral model (requires API key)
3. Choose an OCR option based on your selected parser:
   - **None**: No OCR processing (for documents with selectable text)
   - **Tesseract**: Basic OCR using Tesseract
   - **Advanced**: Enhanced OCR with layout preservation (available with specific parsers)
   - **Plain Text**: For GOT-OCR, extracts raw text without formatting
   - **Formatted Text**: For GOT-OCR, preserves formatting and converts to Markdown
4. Select your desired output format:
   - **Markdown**: Clean, readable markdown format
   - **JSON**: Structured data representation
   - **Text**: Plain text extraction
   - **Document Tags**: XML-like structure tags
5. Click "Convert" to process your document
6. Navigate through pages using the navigation buttons for multi-page documents
7. Download the converted content in your selected format

## Configuration & Error Handling

### 🔧 **Automatic Configuration:**
The application includes intelligent configuration management that:
- Validates API keys and reports availability at startup
- Checks for required dependencies and installs them automatically
- Provides helpful warnings for missing optional components
- Reports which parsers are available based on current configuration

### 🛡️ **Enhanced Error Handling:**
- **User-Friendly Messages**: Clear error descriptions instead of technical stack traces
- **File Validation**: Automatic checking of file size and format compatibility
- **Parser Availability**: Real-time detection of which parsers can be used
- **Graceful Degradation**: Application continues working even if some parsers are unavailable

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
- In Hugging Face Spaces with Stateless GPU, ensure the `spaces` module is imported before any CUDA initialization
- If you see errors about "CUDA must not be initialized in the main process", verify the import order in your app.py
- If you encounter "cannot pickle '_thread.lock' object" errors, this indicates thread locks are being passed to the GPU function
- The GOT-OCR parser has been optimized for ZeroGPU in Stateless GPU environments with proper serialization handling
- For local development, the parser will fall back to CPU processing if GPU is not available

### General Issues
- Check the console logs for error messages
- Ensure all dependencies are installed correctly
- For large documents, try processing fewer pages at a time

## Development Guide

### Project Structure

```
markit_v2/
├── app.py                  # Main application entry point (HF Spaces compatible)
├── run_app.py              # 🆕 Lightweight app launcher for local development
├── setup.sh                # Setup script
├── build.sh                # Build script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .env                    # Environment variables (local development)
├── .gitignore              # Git ignore file
├── .gitattributes          # Git attributes file
├── src/                    # Source code
│   ├── __init__.py         # Package initialization
│   ├── main.py             # Application launcher
│   ├── core/               # Core functionality and utilities
│   │   ├── __init__.py     # Package initialization
│   │   ├── config.py       # 🆕 Centralized configuration management
│   │   ├── exceptions.py   # 🆕 Custom exception hierarchy
│   │   ├── logging_config.py # 🆕 Centralized logging setup
│   │   ├── environment.py  # 🆕 Environment setup and dependency management
│   │   ├── converter.py    # Document conversion orchestrator (refactored)
│   │   ├── parser_factory.py # Parser factory pattern
│   │   └── latex_to_markdown_converter.py # LaTeX conversion utility
│   ├── services/           # Business logic layer
│   │   ├── __init__.py     # Package initialization
│   │   └── document_service.py # 🆕 Document processing service
│   ├── parsers/            # Parser implementations
│   │   ├── __init__.py     # Package initialization
│   │   ├── parser_interface.py # Enhanced parser interface
│   │   ├── parser_registry.py # Parser registry pattern
│   │   ├── markitdown_parser.py # MarkItDown parser (updated)
│   │   ├── got_ocr_parser.py # GOT-OCR parser for images
│   │   ├── mistral_ocr_parser.py # 🆕 Mistral OCR parser
│   │   └── gemini_flash_parser.py # Gemini Flash parser
│   └── ui/                 # User interface layer
│       ├── __init__.py     # Package initialization
│       └── ui.py           # Gradio UI with enhanced error handling
├── documents/              # Documentation and examples (gitignored)
├── tessdata/               # Tesseract OCR data (gitignored)
└── tests/                  # Tests (future)
    └── __init__.py         # Package initialization
```

### 🆕 **New Architecture Components:**
- **Configuration Management**: Centralized API keys, model settings, and app configuration (`src/core/config.py`)
- **Exception Hierarchy**: Proper error handling with specific exception types (`src/core/exceptions.py`)
- **Service Layer**: Business logic separated from UI and core utilities (`src/services/document_service.py`)
- **Environment Management**: Automated dependency checking and setup (`src/core/environment.py`)
- **Enhanced Parser Interface**: Validation, metadata, and cancellation support
- **Lightweight Launcher**: Quick development startup with `run_app.py`
- **Centralized Logging**: Configurable logging system (`src/core/logging_config.py`)

### ZeroGPU Integration Notes

When developing for Hugging Face Spaces with Stateless GPU:

1. Always import the `spaces` module before any CUDA initialization
2. Place all CUDA operations inside functions decorated with `@spaces.GPU()`
3. Ensure only picklable objects are passed to GPU-decorated functions
4. Use wrapper functions to filter out unpicklable objects like thread locks
5. For advanced use cases, consider implementing fallback mechanisms for serialization errors
6. **Add `hf_oauth: true` to your Space's README.md metadata** to mitigate GPU quota limitations
7. Sign in with your Hugging Face account when using the app to utilize your personal GPU quota
8. For extensive GPU usage without quota limitations, a Hugging Face Pro subscription is required

> **Note**: If you're implementing a Space with ZeroGPU on your own, you may encounter quota limitations ("GPU task aborted" errors). These can be mitigated by:
> - Adding `hf_oauth: true` to your Space's metadata (as shown in this Space)
> - Having users sign in with their Hugging Face accounts
> - Upgrading to a Hugging Face Pro subscription for dedicated GPU resources