import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import io

# Import the parser interface and registry
from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry

# Check for MarkItDown availability
try:
    from markitdown import MarkItDown
    from openai import OpenAI
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False
    logging.warning("MarkItDown package not installed. Please install with 'pip install markitdown[all]'")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MarkItDownParser(DocumentParser):
    """
    Parser implementation using MarkItDown for converting various file formats to Markdown.
    """
    
    def __init__(self):
        self.markdown_instance = None
        # Initialize MarkItDown instance
        if HAS_MARKITDOWN:
            try:
                # Check for OpenAI API key for LLM-based image descriptions
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if openai_api_key:
                    client = OpenAI()
                    self.markdown_instance = MarkItDown(
                        enable_plugins=False,
                        llm_client=client, 
                        llm_model="gpt-4o"
                    )
                    logger.info("MarkItDown initialized with OpenAI support for image descriptions")
                else:
                    self.markdown_instance = MarkItDown(enable_plugins=False)
                    logger.info("MarkItDown initialized without OpenAI support")
            except Exception as e:
                logger.error(f"Error initializing MarkItDown: {str(e)}")
                self.markdown_instance = None
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """
        Parse a document and return its content as Markdown.
        
        Args:
            file_path: Path to the document
            ocr_method: OCR method to use (not used in this parser)
            **kwargs: Additional options including cancellation checking
        
        Returns:
            str: Markdown representation of the document
        """
        # Check if MarkItDown is available
        if not HAS_MARKITDOWN or self.markdown_instance is None:
            return "Error: MarkItDown is not available. Please install with 'pip install markitdown[all]'"
            
        # Get cancellation check function from kwargs
        check_cancellation = kwargs.get('check_cancellation', lambda: False)
        
        # Check for cancellation before starting
        if check_cancellation():
            return "Conversion cancelled."
            
        try:
            # Convert the file using the standard instance
            result = self.markdown_instance.convert(file_path)
                
            # Check for cancellation after processing
            if check_cancellation():
                return "Conversion cancelled."
                
            return result.text_content
        except Exception as e:
            logger.error(f"Error converting file with MarkItDown: {str(e)}")
            return f"Error: {str(e)}"
    
    @classmethod
    def get_name(cls) -> str:
        return "MarkItDown (pdf, jpg, png, xlsx --best for xlsx)"
    
    @classmethod
    def get_supported_ocr_methods(cls) -> List[Dict[str, Any]]:
        return [
            {
                "id": "standard",
                "name": "Standard Conversion",
                "default_params": {}
            }
        ]
    
    @classmethod
    def get_description(cls) -> str:
        return "MarkItDown parser for converting various file formats to Markdown"


# Register the parser with the registry if available
if HAS_MARKITDOWN:
    ParserRegistry.register(MarkItDownParser)
    logger.info("MarkItDown parser registered successfully")
else:
    logger.warning("Could not register MarkItDown parser: Package not installed") 