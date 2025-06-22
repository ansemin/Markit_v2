from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
import threading

from src.core.exceptions import ParserError, UnsupportedFileTypeError


class DocumentParser(ABC):
    """Base interface for all document parsers in the system."""
    
    def __init__(self):
        """Initialize the parser."""
        self._cancellation_flag: Optional[threading.Event] = None
    
    def set_cancellation_flag(self, flag: Optional[threading.Event]) -> None:
        """Set the cancellation flag for this parser."""
        self._cancellation_flag = flag
    
    def _check_cancellation(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancellation_flag is not None and self._cancellation_flag.is_set()
    
    @abstractmethod
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """
        Parse a document and return its content.
        
        Args:
            file_path: Path to the document
            ocr_method: OCR method to use (if applicable)
            **kwargs: Additional parser-specific options
            
        Returns:
            str: The parsed content
            
        Raises:
            ParserError: For general parsing errors
            UnsupportedFileTypeError: For unsupported file types
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the display name of this parser"""
        pass
    
    @classmethod
    @abstractmethod
    def get_supported_ocr_methods(cls) -> List[Dict[str, Any]]:
        """
        Return a list of supported OCR methods.
        
        Returns:
            List of dictionaries with keys:
                - id: Unique identifier for the OCR method
                - name: Display name for the OCR method
                - default_params: Default parameters for this OCR method
        """
        pass
    
    @classmethod
    def get_description(cls) -> str:
        """Return a description of this parser"""
        return f"{cls.get_name()} document parser"
    
    @classmethod
    def get_supported_file_types(cls) -> Set[str]:
        """Return a set of supported file extensions (including the dot)."""
        return {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if this parser is available with current configuration."""
        return True
    
    def validate_file(self, file_path: Union[str, Path]) -> None:
        """
        Validate that the file can be processed by this parser.
        
        Args:
            file_path: Path to the file to validate
            
        Raises:
            UnsupportedFileTypeError: If file type is not supported
            ParserError: For other validation errors
        """
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"File not found: {file_path}")
        
        if path.suffix.lower() not in self.get_supported_file_types():
            raise UnsupportedFileTypeError(
                f"File type '{path.suffix}' not supported by {self.get_name()}"
            )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this parser instance."""
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "supported_file_types": list(self.get_supported_file_types()),
            "supported_ocr_methods": self.get_supported_ocr_methods(),
            "available": self.is_available()
        } 