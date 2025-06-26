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
  - Docling: For advanced PDF understanding with table structure recognition + **multi-document processing**
  - GOT-OCR: For image-based OCR with **native LaTeX output** and Mathpix rendering
  - Gemini Flash: For AI-powered text extraction with **advanced multi-document capabilities**
  - Mistral OCR: High-accuracy OCR for PDFs and images with optional *Document Understanding* mode + **multi-document processing**
- **🆕 Intelligent Processing Types**:
  - **Combined**: Merge documents into unified content with duplicate removal
  - **Individual**: Separate sections per document with clear organization
  - **Summary**: Executive overview + detailed analysis of all documents
  - **Comparison**: Cross-document analysis with similarities/differences tables
- Download converted documents as Markdown files

### 🤖 RAG Chat with Documents
- **Chat with your converted documents** using advanced AI
- **🆕 Advanced Retrieval Strategies**: Multiple search methods for optimal results
  - **Similarity Search**: Traditional semantic similarity using embeddings
  - **MMR (Maximal Marginal Relevance)**: Diverse results with reduced redundancy
  - **BM25 Keyword Search**: Traditional keyword-based retrieval
  - **Hybrid Search**: Combines semantic + keyword search for best accuracy
- **Intelligent document retrieval** using vector embeddings
- **🆕 Smart Content-Aware Chunking**: 
  - **Markdown chunking** that preserves tables and code blocks
  - **LaTeX chunking** that preserves mathematical tables, environments, and structures
  - **Automatic format detection** for optimal chunking strategy
- **Streaming chat responses** for real-time interaction
- **Chat history management** with session persistence
- **Usage limits** to prevent abuse on public spaces
- **Powered by Gemini 2.5 Flash** for high-quality responses
- **OpenAI embeddings** for accurate document retrieval
- **🗑️ Clear All Data** button for easy data management in both local and HF Space environments

### 🔍 Query Ranker (NEW!)
- **🆕 Third dedicated tab** for document search and ranking
- **Interactive query search** with real-time document chunk ranking
- **Multiple retrieval methods**: Similarity, MMR, BM25, and Hybrid search
- **Intelligent confidence scoring**: Rank-based confidence levels (High/Medium/Low)
- **Real similarity scores**: Actual ChromaDB similarity scores for similarity search
- **Transparent results**: Clear display of source documents, page numbers, and chunk lengths
- **Adjustable result count**: 1-10 results with responsive slider control
- **Method comparison**: Test different retrieval strategies on the same query
- **Modern card-based UI**: Clean, professional result display with hover effects

### User Interface
- **🆕 Three-tab interface**: Document Converter + Chat + Query Ranker
- **🆕 Unified File Input**: Single interface handles both single and multiple file uploads
- **🆕 Dynamic Processing Options**: Multi-document processing type selector appears automatically
- **🆕 Real-time Validation**: Live feedback on file count, size limits, and processing mode
- **Real-time status monitoring** for RAG system with environment detection
- **Auto-ingestion** of converted documents into chat system
- **Enhanced status display**: Shows vector store document count, chat history files, and environment type
- **Data management controls**: Clear All Data button with comprehensive feedback
- **Filename preservation**: Downloaded files maintain original names (e.g., "example data.pdf" → "example data.md")
- **🆕 Smart Output Naming**: Batch processing creates descriptive filenames (e.g., "Combined_3_Documents_20240125.md")
- **🆕 Consistent modern styling**: All tabs share the same professional design theme
- Clean, responsive UI with modern styling

## Supported Libraries

**MarkItDown** ([Microsoft](https://github.com/microsoft/markitdown)): PDF, Office docs, images, audio, HTML, ZIP files, YouTube URLs, EPubs, and more.

**Docling** ([IBM](https://github.com/DS4SD/docling)): Advanced PDF understanding with table structure recognition, multiple OCR engines, and layout analysis. **Supports multi-document processing** with Gemini-powered summary & comparison.

**Gemini Flash** ([Google](https://deepmind.google/technologies/gemini/)): AI-powered document understanding with **advanced multi-document processing capabilities**, cross-format analysis, and intelligent content synthesis.

**Mistral OCR**: High-accuracy OCR for PDFs and images with optional *Document Understanding* mode. **Supports multi-document processing** with Gemini-powered summary & comparison.

## 🚀 Multi-Document Processing

### **What makes this special?**
Markit v2 introduces **industry-leading multi-document processing** with **three powerful parser options**: Gemini Flash (native multi-document AI), Mistral OCR (high-accuracy with Document Understanding), and Docling (advanced PDF analysis). All support intelligent cross-document analysis.

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
- `CHUNK_SIZE`: Document chunk size for Markdown content (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks for Markdown (default: 200)
- `LATEX_CHUNK_SIZE`: Document chunk size for LaTeX content (default: 1200)
- `LATEX_CHUNK_OVERLAP`: Overlap between chunks for LaTeX (default: 150)
- `MAX_MESSAGES_PER_SESSION`: Chat limit per session (default: 50)
- `MAX_MESSAGES_PER_HOUR`: Chat limit per hour (default: 100)
- `RETRIEVAL_K`: Number of documents to retrieve (default: 4)
- `RAG_MODEL`: Model for RAG chat (default: gemini-2.5-flash)
- `RAG_TEMPERATURE`: Temperature for RAG responses (default: 0.1)
- `RAG_MAX_TOKENS`: Max tokens for RAG responses (default: 4096)

### 🔍 **Advanced Retrieval Configuration:**
- `DEFAULT_RETRIEVAL_METHOD`: Default retrieval strategy (default: similarity)
- `MMR_LAMBDA_MULT`: MMR diversity parameter (default: 0.5)
- `MMR_FETCH_K`: MMR candidate document count (default: 10)
- `HYBRID_SEMANTIC_WEIGHT`: Semantic search weight in hybrid mode (default: 0.7)
- `HYBRID_KEYWORD_WEIGHT`: Keyword search weight in hybrid mode (default: 0.3)
- `BM25_K1`: BM25 term frequency saturation parameter (default: 1.2)
- `BM25_B`: BM25 field length normalization parameter (default: 0.75)

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
6. **For GOT-OCR**: View the LaTeX output with **Mathpix rendering** for proper mathematical and tabular display
7. **For other parsers**: View the Markdown output
8. Download the converted file (.tex for GOT-OCR, .md for others)

#### 📂 **Multi-Document Processing** (NEW!)
1. Go to the **"Document Converter"** tab
2. Upload **2-5 files** (up to 20MB combined)
3. **Processing type selector appears automatically**
4. Choose your processing type:
   - **Combined**: Merge all documents into unified content with smart duplicate removal
   - **Individual**: Keep documents separate with clear section headers
   - **Summary**: Executive overview + detailed analysis of each document
   - **Comparison**: Side-by-side analysis with similarities/differences tables
5. Choose your preferred parser:
   - **Gemini Flash**: Best for advanced cross-document reasoning and native multi-document support
   - **Mistral OCR**: Great for high-accuracy OCR with Document Understanding mode
   - **Docling**: Excellent for PDF table structure + multi-document analysis
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
3. **🆕 Choose your retrieval strategy** for optimal results:
   - **Similarity**: Best for general semantic search
   - **MMR**: Best for diverse, non-repetitive results
   - **Hybrid**: Best overall accuracy (recommended)
4. Ask questions about your converted documents
5. Enjoy real-time streaming responses with document context
6. Use "New Session" to start fresh conversations
7. Use "🗑️ Clear All Data" to remove all documents and chat history
8. Monitor your usage limits in the status panel

### 🔍 Query Ranker (NEW!)
1. Go to the **"Query Ranker"** tab
2. Check the system status to ensure documents are loaded
3. **Enter your search query** in the search box
4. **Choose your retrieval method**:
   - **🎯 Similarity Search**: Semantic similarity with real scores
   - **🔀 MMR (Diverse)**: Diverse results with reduced redundancy
   - **🔍 BM25 (Keywords)**: Traditional keyword-based search
   - **🔗 Hybrid (Recommended)**: Best overall accuracy combining semantic + keyword
5. **Adjust result count** (1-10) using the slider
6. **Review ranked results** with confidence levels and source information
7. **Compare methods** by trying different retrieval strategies on the same query
8. Use results to understand how your documents are chunked and ranked

#### 🔍 **Retrieval Strategy Guide:**
- **For research papers**: Use MMR to get diverse perspectives
- **For technical docs**: Use Hybrid for comprehensive coverage
- **For specific facts**: Use Similarity for targeted results
- **For broad topics**: Use Hybrid for balanced semantic + keyword matching
- **For transparency**: Use Query Ranker to see exactly which chunks are being retrieved

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

## 📄 GOT-OCR LaTeX Processing

Markit v2 features **advanced LaTeX processing** for GOT-OCR results, providing proper mathematical and tabular content handling:

### **🎯 Key Features:**

#### **1. Native LaTeX Output**
- **No LLM conversion**: GOT-OCR returns raw LaTeX directly for maximum accuracy
- **Preserves mathematical structures**: Complex formulas, tables, and equations remain intact
- **.tex file output**: Save files in proper LaTeX format for external use

#### **2. Mathpix Markdown Rendering**
- **Professional display**: Uses Mathpix Markdown library (same as official GOT-OCR demo)
- **Complex table support**: Renders `\begin{tabular}`, `\multirow`, `\multicolumn` properly
- **Mathematical expressions**: Displays LaTeX math with proper formatting
- **Base64 iframe embedding**: Secure, isolated rendering environment

#### **3. RAG-Compatible LaTeX Chunking**
- **LaTeX-aware chunker**: Specialized chunking preserves LaTeX structures
- **Complete table preservation**: Entire `\begin{tabular}...\end{tabular}` blocks stay intact
- **Environment detection**: Maintains `\begin{env}...\end{env}` pairs
- **Intelligent separators**: Uses LaTeX commands (`\section`, `\title`) as break points

#### **4. Enhanced Metadata**
- **Content type tracking**: `content_type: "latex"` for proper handling
- **Structure detection**: Identifies tables, environments, and mathematical content
- **Auto-format detection**: GOT-OCR results automatically use LaTeX chunker

### **🔧 Technical Implementation:**

```javascript
// Mathpix rendering (inspired by official GOT-OCR demo)
const html = window.render(latexContent, {htmlTags: true});

// LaTeX structure preservation
\begin{tabular}{|l|c|c|}
\hline Disability & Participants & Results \\
\hline Blind & 5 & $34.5\%, n=1$ \\
\end{tabular}
```

### **📊 Use Cases:**
- **Research papers**: Mathematical formulas and data tables
- **Scientific documents**: Complex equations and statistical data
- **Financial reports**: Tabular data with calculations
- **Academic content**: Mixed text, math, and structured data

## Credits

- [MarkItDown](https://github.com/microsoft/markitdown) by Microsoft
- [GOT-OCR](https://github.com/stepfun-ai/GOT-OCR-2.0) for image-based OCR
- [Mathpix Markdown](https://github.com/Mathpix/mathpix-markdown-it) for LaTeX rendering
- [Gradio](https://gradio.app/) for the UI framework

---

**Author: Anse Min** | [GitHub](https://github.com/ansemin) | [LinkedIn](https://www.linkedin.com/in/ansemin/)

**Project Links:**
- [GitHub Repository](https://github.com/ansemin/Markit_v2)
- [Hugging Face Space](https://huggingface.co/spaces/Ansemin101/Markit_v2)


## 🔍 Advanced RAG Retrieval Strategies

The system supports **four different retrieval methods** for optimal document search and question answering:

### **1. 🎯 Similarity Search (Default)**
- **How it works**: Semantic similarity using OpenAI embeddings
- **Best for**: General questions and semantic understanding
- **Use case**: "What is the main topic of this document?"
- **Configuration**: `{'k': 4, 'search_type': 'similarity'}`
- **Chunking**: Uses content-aware chunking (Markdown or LaTeX) for optimal structure preservation

### **2. 🔀 MMR (Maximal Marginal Relevance)**  
- **How it works**: Balances relevance with result diversity to reduce redundancy
- **Best for**: Research questions requiring diverse perspectives
- **Use case**: "What are different approaches to transformer architecture?"
- **Configuration**: `{'k': 4, 'fetch_k': 10, 'lambda_mult': 0.5}`
- **Benefits**: Prevents repetitive results, ensures comprehensive coverage

### **3. 🔍 BM25 Keyword Search**
- **How it works**: Traditional keyword-based search with TF-IDF scoring
- **Best for**: Exact term matching and specific factual queries
- **Use case**: "Find mentions of 'attention mechanism' in the documents"
- **Configuration**: `{'k': 4}`
- **Benefits**: Excellent for technical terms and specific concepts

### **4. 🔗 Hybrid Search (Recommended)**
- **How it works**: Combines semantic embeddings + keyword search using ensemble weighting
- **Best for**: Most queries - provides best overall accuracy
- **Use case**: Any complex question benefiting from both semantic and keyword matching
- **Configuration**: `{'k': 4, 'semantic_weight': 0.7, 'keyword_weight': 0.3}`
- **Benefits**: **87.5% hit rate vs 79.2% for similarity-only** (based on LangChain research)

### **🎯 Performance Comparison:**
| Method | Accuracy | Diversity | Speed | Best Use Case |
|--------|----------|-----------|-------|---------------|
| Similarity | Good | Low | Fast | General semantic questions |
| MMR | Good | High | Medium | Research requiring diverse viewpoints |
| BM25 | Medium | Medium | Fast | Exact term/keyword searches |
| **Hybrid** | **Excellent** | **High** | **Medium** | **Most questions (recommended)** |

### **💡 Usage Examples:**

```python
# In your application code
from src.rag.chat_service import rag_chat_service

# Use hybrid search (recommended)
response = rag_chat_service.chat_with_retrieval(
    "How does attention work in transformers?",
    retrieval_method="hybrid",
    retrieval_config={'k': 4, 'semantic_weight': 0.8, 'keyword_weight': 0.2}
)

# Use MMR for diverse research results
response = rag_chat_service.chat_with_retrieval(
    "What are different transformer architectures?", 
    retrieval_method="mmr",
    retrieval_config={'k': 3, 'fetch_k': 10, 'lambda_mult': 0.6}
)
```

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
│       └── ui.py           # 🆕 Gradio UI with three tabs (Converter + Chat + Query Ranker)
├── documents/              # Documentation and examples (gitignored)
├── tessdata/               # Tesseract OCR data (gitignored)
└── tests/                  # 🆕 Test suite for Phase 1 RAG implementation
    ├── __init__.py         # Package initialization
    ├── README.md           # Test documentation and usage guide
    ├── test_implementation_structure.py # Structure validation (no API keys)
    ├── test_retrieval_methods.py # Full functionality testing
    └── test_data_usage.py  # Data usage demonstration
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
- **🆕 Query Ranker Interface**: Dedicated transparency tool for document search and ranking

### 🧠 **RAG System Architecture:**
- **Embeddings Management** (`src/rag/embeddings.py`): OpenAI text-embedding-3-small integration
- **🆕 Smart Content-Aware Chunking** (`src/rag/chunking.py`): 
  - **Unified chunker** supporting both Markdown and LaTeX content
  - **Markdown chunking**: Preserves tables and code blocks as whole units
  - **LaTeX chunking**: Preserves `\begin{tabular}`, mathematical environments, and LaTeX structures
  - **Automatic format detection**: GOT-OCR results → LaTeX chunker, others → Markdown chunker
  - **Enhanced metadata**: Content type tracking and structure detection
- **🆕 Advanced Vector Store** (`src/rag/vector_store.py`): Multi-strategy retrieval system with:
  - **Similarity Search**: Traditional semantic retrieval using embeddings
  - **MMR Support**: Maximal Marginal Relevance for diverse results
  - **BM25 Integration**: Keyword-based search with TF-IDF scoring
  - **Hybrid Retrieval**: Ensemble combining semantic + keyword methods
  - **Chroma database**: Persistent storage with deduplication
- **Chat Memory** (`src/rag/memory.py`): Session management and conversation history
- **🆕 Enhanced Chat Service** (`src/rag/chat_service.py`): Multi-method RAG with Gemini 2.5 Flash
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