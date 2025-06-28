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

A powerful Hugging Face Space that converts various document formats to Markdown and enables intelligent chat with your documents using advanced RAG (Retrieval-Augmented Generation).

## 🎯 System Overview

<div align="center">
<img src="img/Overall%20System%20Workflow%20(Essential).png" alt="Overall System Workflow" width="400">

*Complete workflow from document upload to intelligent RAG chat interaction*
</div>

## ✨ Key Features

### Document Conversion
- Convert PDFs, Office documents, images, and more to Markdown
- **🆕 Multi-Document Processing**: Process up to 5 files simultaneously (20MB combined)
- **5 Powerful Parsers**:
  - **Gemini Flash**: General Purpose + High Accuracy
  - **Mistral OCR**: Fastest Processing
  - **Docling**: Open Source
  - **GOT-OCR**: Document to LaTeX + Open Source  
  - **MarkItDown**: High Accuracy CSV/XML + Open Source
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

<img src="img/Multi-Document%20Processing%20Types%20(Flagship%20Feature).png" alt="Multi-Document Processing Types" width="700">

*Industry-leading multi-document processing with 4 intelligent processing types*

### **Key Capabilities:**
- **📊 Cross-Document Analysis**: Compare and contrast information across different files
- **🔄 Smart Duplicate Removal**: Intelligently merges overlapping content while preserving unique insights
- **📋 Format Intelligence**: Handles mixed file types (PDF + images, Word + Excel, etc.) seamlessly
- **🧠 Contextual Understanding**: Recognizes relationships and patterns across document boundaries

### **Processing Types:**

- **🔗 Combined**: Merge documents into unified content with duplicate removal
- **📑 Individual**: Separate sections per document with clear organization  
- **📈 Summary**: Executive overview + detailed analysis of all documents
- **⚖️ Comparison**: Cross-document analysis with similarities/differences tables

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

## 📖 Usage Guide

### 🎯 Parser Selection

<img src="img/Parser%20Selection%20Guide%20(User-Friendly).png" alt="Parser Selection Guide" width="700">

*Choose the right parser for your specific needs and document types*

### Document Conversion

#### 📄 **Single Document Processing**
1. Upload a single file
2. Choose your preferred parser
3. Select an OCR method based on your chosen parser
4. Click "Convert"
5. Download the converted file (.tex for GOT-OCR, .md for others)

#### 📂 **Multi-Document Processing**
1. Upload **2-5 files** (up to 20MB combined)
2. Choose processing type: Combined, Individual, Summary, or Comparison
3. Select your preferred parser
4. Click "Convert" for intelligent cross-document analysis

### 🤖 RAG Chat & Query System

<img src="img/RAG%20Retrieval%20Strategies%20(Technical%20Highlight).png" alt="RAG Retrieval Strategies" width="700">

*Advanced RAG system with 4 retrieval strategies for optimal document search*

#### **Chat with Documents**
1. Choose your retrieval strategy (Similarity, MMR, BM25, or Hybrid)
2. Ask questions about your converted documents
3. Get real-time streaming responses with document context

#### **Query Ranker**
1. Enter search queries to explore document chunks
2. Compare different retrieval methods
3. View confidence scores and source information

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


## 🔍 Retrieval Strategies

| Method | Best For | Accuracy |
|--------|----------|----------|
| **🎯 Similarity** | General semantic questions | Good |
| **🔀 MMR** | Diverse perspectives | Good |
| **🔍 BM25** | Exact keyword searches | Medium |
| **🔗 Hybrid** | Most queries (recommended) | **Excellent** |

## 💻 Development

### Quick Start
```bash
# Clone repository
git clone https://github.com/ansemin/Markit_v2

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Install dependencies & run
pip install -r requirements.txt
python app.py
```

### Key Technologies
- **Parsers**: Gemini Flash, Mistral OCR, Docling, GOT-OCR, MarkItDown
- **RAG System**: OpenAI embeddings + Chroma vector store + Gemini 2.5 Flash
- **UI Framework**: Gradio with modular component architecture  
- **GPU Support**: ZeroGPU integration for HF Spaces
