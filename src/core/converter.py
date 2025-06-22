import logging
import threading
from typing import Optional, Tuple

from src.core.config import config
from src.core.exceptions import (
    DocumentProcessingError,
    ConversionError,
    ConfigurationError
)
from src.services.document_service import DocumentService

# Import all parsers to ensure they're registered
from src import parsers

# Global document service instance
_document_service = DocumentService()

def set_cancellation_flag(flag: threading.Event) -> None:
    """Set the reference to the cancellation flag from ui.py"""
    _document_service.set_cancellation_flag(flag)

def is_conversion_in_progress() -> bool:
    """Check if conversion is currently in progress"""
    return _document_service.is_conversion_in_progress()

def convert_file(file_path: str, parser_name: str, ocr_method_name: str, output_format: str) -> Tuple[str, Optional[str]]:
    """
    Convert a file using the specified parser and OCR method.
    
    Args:
        file_path: Path to the file
        parser_name: Name of the parser to use
        ocr_method_name: Name of the OCR method to use
        output_format: Output format (Markdown, JSON, Text, Document Tags)
        
    Returns:
        tuple: (content, download_file_path)
    """
    if not file_path:
        return "Please upload a file.", None
    
    try:
        # Use the document service to handle conversion
        content, output_path = _document_service.convert_document(
            file_path=file_path,
            parser_name=parser_name,
            ocr_method_name=ocr_method_name,
            output_format=output_format
        )
        
        return content, output_path
        
    except ConversionError as e:
        # Handle user-friendly conversion errors
        if "cancelled" in str(e).lower():
            return "Conversion cancelled.", None
        return f"Conversion failed: {e}", None
        
    except DocumentProcessingError as e:
        # Handle document processing errors
        return f"Document processing error: {e}", None
        
    except ConfigurationError as e:
        # Handle configuration errors
        return f"Configuration error: {e}", None
        
    except Exception as e:
        # Handle unexpected errors
        logging.error(f"Unexpected error in convert_file: {e}")
        return f"Unexpected error: {e}", None
