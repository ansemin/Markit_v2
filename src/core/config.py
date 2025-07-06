"""
Centralized configuration management for Markit application.
"""
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class APIConfig:
    """Configuration for external API services."""
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load API keys from environment variables."""
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY") 
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")


@dataclass
class OCRConfig:
    """Configuration for OCR-related settings."""
    tesseract_path: Optional[str] = None
    tessdata_path: Optional[str] = None
    default_language: str = "eng"
    
    def __post_init__(self):
        """Load OCR configuration from environment variables."""
        self.tesseract_path = os.getenv("TESSERACT_PATH")
        self.tessdata_path = os.getenv("TESSDATA_PATH", "./tessdata")


@dataclass
class ModelConfig:
    """Configuration for AI model settings."""
    gemini_model: str = "gemini-2.5-flash"
    mistral_model: str = "mistral-ocr-latest"
    got_ocr_model: str = "stepfun-ai/GOT-OCR2_0"
    temperature: float = 0.1
    max_tokens: int = 32768
    
    def __post_init__(self):
        """Load model configuration from environment variables."""
        self.gemini_model = os.getenv("GEMINI_MODEL", self.gemini_model)
        self.mistral_model = os.getenv("MISTRAL_MODEL", self.mistral_model)
        self.got_ocr_model = os.getenv("GOT_OCR_MODEL", self.got_ocr_model)
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", self.temperature))
        self.max_tokens = int(os.getenv("MODEL_MAX_TOKENS", self.max_tokens))


@dataclass
class DoclingConfig:
    """Configuration for Docling parser."""
    artifacts_path: Optional[str] = None
    enable_remote_services: bool = False
    enable_tables: bool = True
    enable_code_enrichment: bool = False
    enable_formula_enrichment: bool = False
    enable_picture_classification: bool = False
    generate_picture_images: bool = False
    ocr_cpu_threads: int = 4
    
    def __post_init__(self):
        """Load Docling configuration from environment variables."""
        self.artifacts_path = os.getenv("DOCLING_ARTIFACTS_PATH")
        self.enable_remote_services = os.getenv("DOCLING_ENABLE_REMOTE_SERVICES", "false").lower() == "true"
        self.enable_tables = os.getenv("DOCLING_ENABLE_TABLES", "true").lower() == "true"
        self.enable_code_enrichment = os.getenv("DOCLING_ENABLE_CODE_ENRICHMENT", "false").lower() == "true"
        self.enable_formula_enrichment = os.getenv("DOCLING_ENABLE_FORMULA_ENRICHMENT", "false").lower() == "true"
        self.enable_picture_classification = os.getenv("DOCLING_ENABLE_PICTURE_CLASSIFICATION", "false").lower() == "true"
        self.generate_picture_images = os.getenv("DOCLING_GENERATE_PICTURE_IMAGES", "false").lower() == "true"
        self.ocr_cpu_threads = int(os.getenv("OMP_NUM_THREADS", self.ocr_cpu_threads))


@dataclass
class RAGConfig:
    """Configuration for RAG (Retrieval-Augmented Generation) functionality."""
    # Vector store settings
    vector_store_path: str = "./data/vector_store"
    collection_name: str = "markit_documents"
    
    # Chat history settings
    chat_history_path: str = "./data/chat_history"
    
    # Embedding settings
    embedding_model: str = "models/text-embedding-004"
    embedding_chunk_size: int = 1000
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Chat limits
    max_messages_per_session: int = 50
    max_messages_per_hour: int = 100
    
    # Retrieval settings
    retrieval_k: int = 4
    retrieval_score_threshold: float = 0.5
    
    # LLM settings for RAG
    rag_model: str = "gemini-2.5-flash"
    rag_temperature: float = 0.1
    rag_max_tokens: int = 32768
    
    def __post_init__(self):
        """Load RAG configuration from environment variables."""
        # For HF Spaces, ensure data directories are created
        if os.getenv("SPACE_ID"):  # HF Spaces environment
            base_data_path = "/tmp/data" if not os.access("./data", os.W_OK) else "./data"
            self.vector_store_path = os.getenv("VECTOR_STORE_PATH", f"{base_data_path}/vector_store")
            self.chat_history_path = os.getenv("CHAT_HISTORY_PATH", f"{base_data_path}/chat_history")
        else:
            self.vector_store_path = os.getenv("VECTOR_STORE_PATH", self.vector_store_path)
            self.chat_history_path = os.getenv("CHAT_HISTORY_PATH", self.chat_history_path)
        
        self.collection_name = os.getenv("VECTOR_STORE_COLLECTION", self.collection_name)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.embedding_chunk_size = int(os.getenv("EMBEDDING_CHUNK_SIZE", self.embedding_chunk_size))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", self.chunk_size))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", self.chunk_overlap))
        self.max_messages_per_session = int(os.getenv("MAX_MESSAGES_PER_SESSION", self.max_messages_per_session))
        self.max_messages_per_hour = int(os.getenv("MAX_MESSAGES_PER_HOUR", self.max_messages_per_hour))
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", self.retrieval_k))
        self.retrieval_score_threshold = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", self.retrieval_score_threshold))
        self.rag_model = os.getenv("RAG_MODEL", self.rag_model)
        self.rag_temperature = float(os.getenv("RAG_TEMPERATURE", self.rag_temperature))
        self.rag_max_tokens = int(os.getenv("RAG_MAX_TOKENS", self.rag_max_tokens))


@dataclass
class AppConfig:
    """Main application configuration."""
    debug: bool = False
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: tuple = (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".tex", ".xlsx", ".docx", ".pptx", ".html", ".xhtml", ".md", ".csv")
    temp_dir: str = "./temp"
    
    # Multi-document batch processing settings
    max_batch_files: int = 5
    max_batch_size: int = 20 * 1024 * 1024  # 20MB combined
    batch_processing_types: tuple = ("combined", "individual", "summary", "comparison")
    
    def __post_init__(self):
        """Load application configuration from environment variables."""
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", self.max_file_size))
        self.temp_dir = os.getenv("TEMP_DIR", self.temp_dir)
        
        # Load batch processing configuration
        self.max_batch_files = int(os.getenv("MAX_BATCH_FILES", self.max_batch_files))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", self.max_batch_size))


class Config:
    """Main configuration container."""
    
    def __init__(self):
        self.api = APIConfig()
        self.ocr = OCRConfig()
        self.model = ModelConfig()
        self.docling = DoclingConfig()
        self.app = AppConfig()
        self.rag = RAGConfig()
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check API keys
        if not self.api.google_api_key:
            validation_results["warnings"].append("Google API key not found - Gemini parser will be unavailable")
        
        if not self.api.mistral_api_key:
            validation_results["warnings"].append("Mistral API key not found - Mistral parser will be unavailable")
        
        # Check RAG dependencies
        if not self.api.google_api_key:
            validation_results["warnings"].append("Google API key not found - RAG embeddings will be unavailable")
        
        if not self.api.google_api_key:
            validation_results["warnings"].append("Google API key not found - RAG chat will be unavailable")
        
        # Check tesseract setup
        if not self.ocr.tesseract_path and not os.path.exists("/usr/bin/tesseract"):
            validation_results["warnings"].append("Tesseract not found in system PATH - OCR functionality may be limited")
        
        # Check temp directory
        try:
            os.makedirs(self.app.temp_dir, exist_ok=True)
        except Exception as e:
            validation_results["errors"].append(f"Cannot create temp directory {self.app.temp_dir}: {e}")
            validation_results["valid"] = False
        
        # Check RAG directories
        try:
            os.makedirs(self.rag.vector_store_path, exist_ok=True)
            os.makedirs(self.rag.chat_history_path, exist_ok=True)
        except Exception as e:
            validation_results["errors"].append(f"Cannot create RAG directories: {e}")
            validation_results["valid"] = False
        
        return validation_results
    
    def get_available_parsers(self) -> list:
        """Get list of available parsers based on current configuration."""
        available = ["markitdown"]  # Always available
        
        if self.api.google_api_key:
            available.append("gemini_flash")
        
        if self.api.mistral_api_key:
            available.append("mistral_ocr")
        
        # GOT-OCR is available if we have GPU or can use ZeroGPU
        available.append("got_ocr")
        
        # Docling is available if package is installed
        try:
            import docling
            available.append("docling")
        except ImportError:
            pass
        
        return available


# Global configuration instance
config = Config()