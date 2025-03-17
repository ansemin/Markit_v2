from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


class DocumentParser(ABC):
    """Base interface for all document parsers in the system."""
    
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