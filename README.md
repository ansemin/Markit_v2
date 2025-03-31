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

You can enhance the functionality by setting these environment variables:

- `OPENAI_API_KEY`: Enables AI-based image descriptions in MarkItDown
- `GOOGLE_API_KEY`: Used for Gemini Flash parser and LaTeX to Markdown conversion

## Usage

1. Select a file to upload
2. Choose "MarkItDown" as the parser
3. Select "Standard Conversion"
4. Click "Convert"
5. View the Markdown output and download the converted file

## Local Development

1. Clone the repository
2. Create a `.env` file based on `.env.example`
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   python app.py
   ```

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
  - **PyPdfium**: Fast PDF parsing using the PDFium engine
  - **Docling**: Advanced document structure analysis
  - **Gemini Flash**: AI-powered conversion using Google's Gemini API
  - **GOT-OCR**: State-of-the-art OCR model for images (JPG/PNG only) with plain text and formatted text options
- **OCR Integration**: Extract text from images and scanned documents using Tesseract OCR
- **Interactive UI**: User-friendly Gradio interface with page navigation for large documents
- **AI-Powered Chat**: Interact with your documents using AI to ask questions about content
- **ZeroGPU Support**: Optimized for Hugging Face Spaces with Stateless GPU environments

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
   - **PyPdfium**: Best for standard PDFs with selectable text
   - **Docling**: Best for complex document layouts
   - **Gemini Flash**: Best for AI-powered conversions (requires API key)
   - **GOT-OCR**: Best for high-quality OCR on images (JPG/PNG only)
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
│   │   ├── got_ocr_parser.py # GOT-OCR parser for images
│   │   └── pypdfium_parser.py # PyPDFium parser
│   ├── ui/                 # User interface
│   │   ├── __init__.py     # Package initialization
│   │   └── ui.py           # Gradio UI implementation
│   └── services/           # External services
│       └── __init__.py     # Package initialization
└── tests/                  # Tests
    └── __init__.py         # Package initialization
```

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