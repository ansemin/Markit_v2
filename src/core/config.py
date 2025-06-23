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
class AppConfig:
    """Main application configuration."""
    debug: bool = False
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: tuple = (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".tex", ".xlsx", ".docx", ".pptx", ".html", ".xhtml", ".md", ".csv")
    temp_dir: str = "./temp"
    
    def __post_init__(self):
        """Load application configuration from environment variables."""
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", self.max_file_size))
        self.temp_dir = os.getenv("TEMP_DIR", self.temp_dir)


class Config:
    """Main configuration container."""
    
    def __init__(self):
        self.api = APIConfig()
        self.ocr = OCRConfig()
        self.model = ModelConfig()
        self.docling = DoclingConfig()
        self.app = AppConfig()
    
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
        
        # Check tesseract setup
        if not self.ocr.tesseract_path and not os.path.exists("/usr/bin/tesseract"):
            validation_results["warnings"].append("Tesseract not found in system PATH - OCR functionality may be limited")
        
        # Check temp directory
        try:
            os.makedirs(self.app.temp_dir, exist_ok=True)
        except Exception as e:
            validation_results["errors"].append(f"Cannot create temp directory {self.app.temp_dir}: {e}")
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