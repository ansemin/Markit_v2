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

# Document to Markdown Converter with RAG Chat

A Hugging Face Space that converts various document formats to Markdown and lets you chat with your documents using RAG (Retrieval-Augmented Generation)!

## ✨ Key Features

### Document Conversion
- Convert PDFs, Office documents, images, and more to Markdown
- **🆕 Multi-Document Processing**: Process up to 5 files simultaneously (20MB combined)
- Multiple parser options:
  - MarkItDown: For comprehensive document conversion
  - Docling: For advanced PDF understanding with table structure recognition
  - GOT-OCR: For image-based OCR with LaTeX support
  - Gemini Flash: For AI-powered text extraction with **advanced multi-document capabilities**
- **🆕 Intelligent Processing Types**:
  - **Combined**: Merge documents into unified content with duplicate removal
  - **Individual**: Separate sections per document with clear organization
  - **Summary**: Executive overview + detailed analysis of all documents
  - **Comparison**: Cross-document analysis with similarities/differences tables
- Download converted documents as Markdown files

### 🤖 RAG Chat with Documents
- **Chat with your converted documents** using advanced AI
- **Intelligent document retrieval** using vector embeddings
- **Markdown-aware chunking** that preserves tables and code blocks
- **Streaming chat responses** for real-time interaction
- **Chat history management** with session persistence
- **Usage limits** to prevent abuse on public spaces
- **Powered by Gemini 2.5 Flash** for high-quality responses
- **OpenAI embeddings** for accurate document retrieval
- **🗑️ Clear All Data** button for easy data management in both local and HF Space environments

### User Interface
- **Dual-tab interface**: Document Converter + Chat
- **🆕 Unified File Input**: Single interface handles both single and multiple file uploads
- **🆕 Dynamic Processing Options**: Multi-document processing type selector appears automatically
- **🆕 Real-time Validation**: Live feedback on file count, size limits, and processing mode
- **Real-time status monitoring** for RAG system with environment detection
- **Auto-ingestion** of converted documents into chat system
- **Enhanced status display**: Shows vector store document count, chat history files, and environment type
- **Data management controls**: Clear All Data button with comprehensive feedback
- **Filename preservation**: Downloaded files maintain original names (e.g., "example data.pdf" → "example data.md")
- **🆕 Smart Output Naming**: Batch processing creates descriptive filenames (e.g., "Combined_3_Documents_20240125.md")
- Clean, responsive UI with modern styling

## Supported Libraries

**MarkItDown** ([Microsoft](https://github.com/microsoft/markitdown)): PDF, Office docs, images, audio, HTML, ZIP files, YouTube URLs, EPubs, and more.

**Docling** ([IBM](https://github.com/DS4SD/docling)): Advanced PDF understanding with table structure recognition, multiple OCR engines, and layout analysis.

**Gemini Flash** ([Google](https://deepmind.google/technologies/gemini/)): AI-powered document understanding with **advanced multi-document processing capabilities**, cross-format analysis, and intelligent content synthesis.

## 🚀 Multi-Document Processing

### **What makes this special?**
Markit v2 introduces **industry-leading multi-document processing** powered by Google's Gemini Flash 2.5, enabling intelligent analysis across multiple documents simultaneously.

### **Key Capabilities:**
- **📊 Cross-Document Analysis**: Compare and contrast information across different files
- **🔄 Smart Duplicate Removal**: Intelligently merges overlapping content while preserving unique insights
- **📋 Format Intelligence**: Handles mixed file types (PDF + images, Word + Excel, etc.) seamlessly
- **🧠 Contextual Understanding**: Recognizes relationships and patterns across document boundaries
- **⚡ Single API Call Processing**: Efficient batch processing using Gemini's native multi-document support

### **Processing Types Explained:**

#### 🔗 **Combined Processing**
- **Purpose**: Create one unified, cohesive document from multiple sources
- **Best for**: Related documents that should be read as one complete resource
- **Intelligence**: Removes redundant information while preserving all critical content
- **Example**: Merge project proposal + budget + timeline into one comprehensive document

#### 📑 **Individual Processing**  
- **Purpose**: Convert each document separately but organize them in one output
- **Best for**: Different documents you want in one place for easy reference
- **Intelligence**: Maintains original structure while creating clear organization
- **Example**: Meeting agenda + presentation + notes → organized sections

#### 📈 **Summary Processing**
- **Purpose**: Executive overview + detailed analysis
- **Best for**: Complex document sets needing high-level insights
- **Intelligence**: Cross-document pattern recognition and key insight extraction
- **Example**: Research papers → executive summary + detailed analysis of each paper

#### ⚖️ **Comparison Processing**
- **Purpose**: Analyze differences, similarities, and relationships
- **Best for**: Multiple proposals, document versions, or conflicting sources
- **Intelligence**: Creates comparison tables and identifies discrepancies/alignments
- **Example**: Contract versions → side-by-side analysis with change identification

### **Technical Advantages:**
- **Native Multimodal Support**: Processes text + images in same workflow
- **Advanced Reasoning**: Understands context and relationships between documents
- **Efficient Processing**: Single Gemini API call vs. multiple individual calls
- **Format Agnostic**: Works across all supported file types seamlessly

## Environment Variables

The application uses centralized configuration management. You can enhance functionality by setting these environment variables:

### 🔑 **API Keys:**
- `GOOGLE_API_KEY`: Used for Gemini Flash parser, LaTeX conversion, and **RAG chat functionality**
- `OPENAI_API_KEY`: Enables AI-based image descriptions in MarkItDown and **vector embeddings for RAG**
- `MISTRAL_API_KEY`: For Mistral OCR parser (if available)

### ⚙️ **Configuration Options:**
- `DEBUG`: Set to `true` for debug mode with verbose logging
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: 10MB)
- `MAX_BATCH_FILES`: Maximum files for multi-document processing (default: 5)
- `MAX_BATCH_SIZE`: Maximum combined size for batch processing (default: 20MB)
- `TEMP_DIR`: Directory for temporary files (default: ./temp)
- `TESSERACT_PATH`: Custom path to Tesseract executable
- `TESSDATA_PATH`: Path to Tesseract language data

### 🔧 **Docling Configuration:**
- `DOCLING_ARTIFACTS_PATH`: Path to pre-downloaded Docling models for offline use
- `DOCLING_ENABLE_REMOTE_SERVICES`: Enable remote vision model services (default: false)
- `DOCLING_ENABLE_TABLES`: Enable table structure recognition (default: true)
- `DOCLING_ENABLE_CODE_ENRICHMENT`: Enable code block enrichment (default: false)
- `DOCLING_ENABLE_FORMULA_ENRICHMENT`: Enable formula understanding (default: false)
- `DOCLING_ENABLE_PICTURE_CLASSIFICATION`: Enable picture classification (default: false)
- `DOCLING_GENERATE_PICTURE_IMAGES`: Generate picture images during processing (default: false)
- `OMP_NUM_THREADS`: Number of CPU threads for OCR processing (default: 4)

### 🤖 **Model Configuration:**
- `GEMINI_MODEL`: Gemini model to use (default: gemini-1.5-flash)
- `MISTRAL_MODEL`: Mistral model to use (default: pixtral-12b-2409)
- `GOT_OCR_MODEL`: GOT-OCR model to use (default: stepfun-ai/GOT-OCR2_0)
- `MODEL_TEMPERATURE`: Model temperature for AI responses (default: 0.1)
- `MODEL_MAX_TOKENS`: Maximum tokens for AI responses (default: 4096)

### 🧠 **RAG Configuration:**
- `VECTOR_STORE_PATH`: Path for vector database storage (default: ./data/vector_store)
- `CHAT_HISTORY_PATH`: Path for chat history storage (default: ./data/chat_history)
- `EMBEDDING_MODEL`: OpenAI embedding model (default: text-embedding-3-small)
- `CHUNK_SIZE`: Document chunk size for RAG (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MAX_MESSAGES_PER_SESSION`: Chat limit per session (default: 50)
- `MAX_MESSAGES_PER_HOUR`: Chat limit per hour (default: 100)
- `RETRIEVAL_K`: Number of documents to retrieve (default: 4)
- `RAG_MODEL`: Model for RAG chat (default: gemini-2.5-flash)
- `RAG_TEMPERATURE`: Temperature for RAG responses (default: 0.1)
- `RAG_MAX_TOKENS`: Max tokens for RAG responses (default: 4096)

## Usage

### Document Conversion

#### 📄 **Single Document Processing**
1. Go to the **"Document Converter"** tab
2. Upload a single file
3. Choose your preferred parser:
   - **"MarkItDown"** for comprehensive document conversion
   - **"Docling"** for advanced PDF understanding and table extraction
   - **"Gemini Flash"** for AI-powered text extraction
4. Select an OCR method based on your chosen parser
5. Click "Convert"
6. View the Markdown output and download the converted file

#### 📂 **Multi-Document Processing** (NEW!)
1. Go to the **"Document Converter"** tab
2. Upload **2-5 files** (up to 20MB combined)
3. **Processing type selector appears automatically**
4. Choose your processing type:
   - **Combined**: Merge all documents into unified content with smart duplicate removal
   - **Individual**: Keep documents separate with clear section headers
   - **Summary**: Executive overview + detailed analysis of each document
   - **Comparison**: Side-by-side analysis with similarities/differences tables
5. Choose your preferred parser (recommend **Gemini Flash** for best multi-document results)
6. Click "Convert"
7. Get intelligent cross-document analysis and download enhanced output

#### 💡 **Multi-Document Tips**
- **Mixed file types work great**: Upload PDF + images, Word docs + PDFs, etc.
- **Gemini Flash excels at**: Cross-document reasoning, duplicate detection, and format analysis
- **Perfect for**: Comparing document versions, analyzing related reports, consolidating research
- **Real-time validation**: UI shows file count, size limits, and processing mode

#### 🤖 **RAG Integration**
- **All converted documents are automatically added to the RAG system** for chat functionality
- Multi-document processing creates richer context for chat interactions

### 🤖 Chat with Documents
1. Go to the **"Chat with Documents"** tab
2. Check the system status to ensure RAG components are ready
3. Ask questions about your converted documents
4. Enjoy real-time streaming responses with document context
5. Use "New Session" to start fresh conversations
6. Use "🗑️ Clear All Data" to remove all documents and chat history
7. Monitor your usage limits in the status panel

## Local Development

### 🚀 **Quick Start:**
1. Clone the repository
2. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   MISTRAL_API_KEY=your_mistral_api_key_here
   DEBUG=true
   
   # RAG Configuration (optional - uses defaults if not set)
   MAX_MESSAGES_PER_SESSION=50
   MAX_MESSAGES_PER_HOUR=100
   CHUNK_SIZE=1000
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
   
   # For testing with clean data
   python run_app.py --clear-data-and-run
   
   # Show all available options
   python run_app.py --help
   ```

### 🧹 **Data Management:**

**Two ways to clear data:**

1. **Command-line** (for development):
   - `python run_app.py --clear-data-and-run` - Clear data then start app
   - `python run_app.py --clear-data` - Clear data and exit

2. **In-app UI** (for users):
   - Go to "Chat with Documents" tab → Click "🗑️ Clear All Data" button
   - Automatically detects environment (local vs HF Space)
   - Provides detailed feedback and starts new session

**What gets cleared:**
- `data/chat_history/*` - All saved chat sessions  
- `data/vector_store/*` - All document embeddings and vector database

### 🧪 **Development Features:**
- **Automatic Environment Setup**: Dependencies are checked and installed automatically
- **Configuration Validation**: Startup validation reports missing API keys and configuration issues
- **Enhanced Error Messages**: Detailed error reporting for debugging
- **Centralized Logging**: Configurable logging levels and output formats

## Credits

- [MarkItDown](https://github.com/microsoft/markitdown) by Microsoft
- [GOT-OCR](https://github.com/stepfun-ai/GOT-OCR-2.0) for image-based OCR
- [Gradio](https://gradio.app/) for the UI framework

---

**Author: Anse Min** | [GitHub](https://github.com/ansemin) | [LinkedIn](https://www.linkedin.com/in/ansemin/)

**Project Links:**
- [GitHub Repository](https://github.com/ansemin/Markit_v2)
- [Hugging Face Space](https://huggingface.co/spaces/Ansemin101/Markit_v2)


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
│   │   ├── config.py       # 🆕 Centralized configuration management (with RAG settings)
│   │   ├── exceptions.py   # 🆕 Custom exception hierarchy
│   │   ├── logging_config.py # 🆕 Centralized logging setup
│   │   ├── environment.py  # 🆕 Environment setup and dependency management
│   │   ├── converter.py    # Document conversion orchestrator (refactored)
│   │   ├── parser_factory.py # Parser factory pattern
│   │   └── latex_to_markdown_converter.py # LaTeX conversion utility
│   ├── services/           # Business logic layer
│   │   ├── __init__.py     # Package initialization
│   │   ├── document_service.py # 🆕 Document processing service
│   │   └── data_clearing_service.py # 🆕 Data management and clearing service
│   ├── parsers/            # Parser implementations
│   │   ├── __init__.py     # Package initialization
│   │   ├── parser_interface.py # Enhanced parser interface
│   │   ├── parser_registry.py # Parser registry pattern
│   │   ├── markitdown_parser.py # MarkItDown parser (updated)
│   │   ├── docling_parser.py # 🆕 Docling parser with advanced PDF understanding
│   │   ├── got_ocr_parser.py # GOT-OCR parser for images
│   │   ├── mistral_ocr_parser.py # 🆕 Mistral OCR parser
│   │   └── gemini_flash_parser.py # 🆕 Enhanced Gemini Flash parser with multi-document processing
│   ├── rag/                # 🆕 RAG (Retrieval-Augmented Generation) system
│   │   ├── __init__.py     # Package initialization
│   │   ├── embeddings.py   # OpenAI embedding model management
│   │   ├── chunking.py     # Markdown-aware document chunking
│   │   ├── vector_store.py # Chroma vector database management
│   │   ├── memory.py       # Chat history and session management
│   │   ├── chat_service.py # RAG chat service with Gemini 2.5 Flash
│   │   └── ingestion.py    # Document ingestion pipeline
│   └── ui/                 # User interface layer
│       ├── __init__.py     # Package initialization
│       └── ui.py           # Gradio UI with dual tabs (Converter + Chat)
├── documents/              # Documentation and examples (gitignored)
├── tessdata/               # Tesseract OCR data (gitignored)
└── tests/                  # Tests (future)
    └── __init__.py         # Package initialization
```

### 🆕 **New Architecture Components:**
- **Configuration Management**: Centralized API keys, model settings, and app configuration (`src/core/config.py`)
- **Exception Hierarchy**: Proper error handling with specific exception types (`src/core/exceptions.py`)
- **Service Layer**: Business logic separated from UI and core utilities (`src/services/document_service.py`)
- **Data Management Service**: Comprehensive data clearing functionality (`src/services/data_clearing_service.py`)
- **Environment Management**: Automated dependency checking and setup (`src/core/environment.py`)
- **Enhanced Parser Interface**: Validation, metadata, and cancellation support
- **Lightweight Launcher**: Quick development startup with `run_app.py`
- **Centralized Logging**: Configurable logging system (`src/core/logging_config.py`)
- **🆕 RAG System**: Complete RAG implementation with vector search and chat capabilities

### 🧠 **RAG System Architecture:**
- **Embeddings Management** (`src/rag/embeddings.py`): OpenAI text-embedding-3-small integration
- **Markdown-Aware Chunking** (`src/rag/chunking.py`): Preserves tables and code blocks as whole units
- **Vector Store** (`src/rag/vector_store.py`): Chroma database with persistent storage and deduplication
- **Chat Memory** (`src/rag/memory.py`): Session management and conversation history
- **Chat Service** (`src/rag/chat_service.py`): Streaming RAG responses with Gemini 2.5 Flash
- **Document Ingestion** (`src/rag/ingestion.py`): Automated pipeline with intelligent duplicate handling
- **Usage Limiting**: Anti-abuse measures for public deployment
- **Auto-Ingestion**: Seamless integration with document conversion workflow

### 🗑️ **Data Management & Deduplication:**
- **File Hash-Based Deduplication**: Uses SHA-256 hashes of original file content to prevent duplicates
- **Chroma Where Filter Integration**: Persistent duplicate detection using vector store metadata queries
- **Automatic Document Replacement**: When same file is uploaded again, old version is replaced with new one
- **Cross-Environment Data Clearing**: Works seamlessly in both local development and HF Space environments
- **Environment-Aware Path Resolution**: Automatically detects and uses correct data paths (`./data/*` vs `/tmp/data/*`)
- **Comprehensive Status Reporting**: Real-time display of vector store documents, chat history files, and environment type
- **Safe Clearing Operations**: Graceful error handling with detailed feedback on clearing operations

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