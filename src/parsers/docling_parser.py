import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
import tempfile

# Import the parser interface and registry
from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry
from src.core.exceptions import DocumentProcessingError, ParserError

# Check for Docling availability
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TesseractOcrOptions
    from docling.document_converter import PdfFormatOption
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False
    logging.warning("Docling package not installed. Please install with 'pip install docling'")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DoclingParser(DocumentParser):
    """
    Parser implementation using Docling for converting documents to Markdown.
    Supports advanced PDF understanding, OCR, and multiple document formats.
    """
    
    def __init__(self):
        super().__init__()  # Initialize the base class (including _cancellation_flag)
        self.converter = None
        
        # Initialize Docling converter
        if HAS_DOCLING:
            try:
                # Create default converter instance
                self.converter = DocumentConverter()
                logger.info("Docling initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Docling: {str(e)}")
                self.converter = None
    
    def _create_converter_with_options(self, ocr_method: str, **kwargs) -> DocumentConverter:
        """Create a DocumentConverter with specific OCR options."""
        pipeline_options = PdfPipelineOptions()
        
        # Enable OCR by default
        pipeline_options.do_ocr = True
        
        # Configure OCR method
        if ocr_method == "docling_tesseract":
            pipeline_options.ocr_options = TesseractOcrOptions()
        elif ocr_method == "docling_easyocr":
            pipeline_options.ocr_options = EasyOcrOptions()
        else:  # Default to EasyOCR
            pipeline_options.ocr_options = EasyOcrOptions()
        
        # Configure advanced features
        pipeline_options.do_table_structure = kwargs.get('enable_tables', True)
        pipeline_options.do_code_enrichment = kwargs.get('enable_code_enrichment', False)
        pipeline_options.do_formula_enrichment = kwargs.get('enable_formula_enrichment', False)
        pipeline_options.do_picture_classification = kwargs.get('enable_picture_classification', False)
        pipeline_options.generate_picture_images = kwargs.get('generate_picture_images', False)
        
        # Create converter with options
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        return converter
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """
        Parse a document and return its content as Markdown.
        
        Args:
            file_path: Path to the document
            ocr_method: OCR method to use ('docling_default', 'docling_tesseract', 'docling_easyocr')
            **kwargs: Additional options for Docling processing
        
        Returns:
            str: Markdown representation of the document
        """
        # Validate file first
        self.validate_file(file_path)
        
        # Check if Docling is available
        if not HAS_DOCLING or self.converter is None:
            raise ParserError("Docling is not available. Please install with 'pip install docling'")
        
        # Check for cancellation before starting
        if self._check_cancellation():
            raise DocumentProcessingError("Conversion cancelled")
        
        try:
            # Use method-specific converter if OCR method is specified
            if ocr_method and ocr_method != "docling_default":
                converter = self._create_converter_with_options(ocr_method, **kwargs)
            else:
                converter = self.converter
            
            # Convert the document
            result = converter.convert(str(file_path))
            
            # Check for cancellation after processing
            if self._check_cancellation():
                raise DocumentProcessingError("Conversion cancelled")
            
            # Export to markdown
            markdown_content = result.document.export_to_markdown()
            
            return markdown_content
            
        except Exception as e:
            logger.error(f"Error converting file with Docling: {str(e)}")
            raise DocumentProcessingError(f"Docling conversion failed: {str(e)}")
    
    @classmethod
    def get_name(cls) -> str:
        return "Docling (PDF, Images, DOCX, XLSX - Advanced PDF Understanding)"
    
    @classmethod
    def get_supported_file_types(cls) -> Set[str]:
        """Return a set of supported file extensions."""
        return {
            # PDF files
            ".pdf",
            # Image files
            ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
            # Office documents
            ".docx", ".xlsx", ".pptx",
            # Web and markup
            ".html", ".xhtml", ".md",
            # Other formats
            ".csv"
        }
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if this parser is available."""
        return HAS_DOCLING
    
    @classmethod
    def get_supported_ocr_methods(cls) -> List[Dict[str, Any]]:
        """Return list of supported OCR methods."""
        methods = [
            {
                "id": "docling_default",
                "name": "Docling Default (EasyOCR)",
                "default_params": {
                    "enable_tables": True,
                    "enable_code_enrichment": False,
                    "enable_formula_enrichment": False,
                    "enable_picture_classification": False,
                    "generate_picture_images": False
                }
            },
            {
                "id": "docling_easyocr",
                "name": "Docling EasyOCR",
                "default_params": {
                    "enable_tables": True,
                    "enable_code_enrichment": False,
                    "enable_formula_enrichment": False,
                    "enable_picture_classification": False,
                    "generate_picture_images": False
                }
            }
        ]
        
        # Add Tesseract method if available (requires system installation)
        try:
            import subprocess
            subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
            methods.append({
                "id": "docling_tesseract",
                "name": "Docling Tesseract OCR",
                "default_params": {
                    "enable_tables": True,
                    "enable_code_enrichment": False,
                    "enable_formula_enrichment": False,
                    "enable_picture_classification": False,
                    "generate_picture_images": False
                }
            })
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.debug("Tesseract not available on system")
        
        return methods
    
    @classmethod
    def get_description(cls) -> str:
        return "Docling parser with advanced PDF understanding, table structure recognition, and multiple OCR engines"


# Register the parser with the registry if available
if HAS_DOCLING:
    ParserRegistry.register(DoclingParser)
    logger.info("Docling parser registered successfully")
else:
    logger.warning("Could not register Docling parser: Package not installed")