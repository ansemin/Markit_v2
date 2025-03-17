from typing import Dict, List, Type, Any, Optional
from src.parsers.parser_interface import DocumentParser


class ParserRegistry:
    """Central registry for all document parsers in the system."""
    
    _parsers: Dict[str, Type[DocumentParser]] = {}
    
    @classmethod
    def register(cls, parser_class: Type[DocumentParser]) -> None:
        """
        Register a parser with the system.
        
        Args:
            parser_class: The parser class to register
        """
        parser_name = parser_class.get_name()
        cls._parsers[parser_name] = parser_class
        print(f"Registered parser: {parser_name}")
    
    @classmethod
    def get_available_parsers(cls) -> Dict[str, Type[DocumentParser]]:
        """Return all registered parsers"""
        return cls._parsers
    
    @classmethod
    def get_parser_class(cls, name: str) -> Optional[Type[DocumentParser]]:
        """Get a specific parser class by name"""
        return cls._parsers.get(name)
    
    @classmethod
    def get_parser_names(cls) -> List[str]:
        """Get a list of all registered parser names"""
        return list(cls._parsers.keys())
    
    @classmethod
    def get_ocr_options(cls, parser_name: str) -> List[str]:
        """
        Get OCR methods supported by a parser.
        
        Args:
            parser_name: Name of the parser
            
        Returns:
            List of OCR method display names
        """
        parser_class = cls.get_parser_class(parser_name)
        if not parser_class:
            return []
        
        return [method["name"] for method in parser_class.get_supported_ocr_methods()]
    
    @classmethod
    def get_ocr_method_id(cls, parser_name: str, ocr_display_name: str) -> Optional[str]:
        """
        Get the internal ID for an OCR method based on its display name.
        
        Args:
            parser_name: Name of the parser
            ocr_display_name: Display name of the OCR method
            
        Returns:
            Internal ID of the OCR method or None if not found
        """
        parser_class = cls.get_parser_class(parser_name)
        if not parser_class:
            return None
        
        for method in parser_class.get_supported_ocr_methods():
            if method["name"] == ocr_display_name:
                return method["id"]
        
        return None 