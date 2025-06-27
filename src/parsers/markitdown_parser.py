import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
import io

# Import the parser interface and registry
from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry
from src.core.exceptions import DocumentProcessingError, ParserError

# Check for MarkItDown availability
try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False
    logging.warning("MarkItDown package not installed. Please install with 'pip install markitdown[all]'")

# Import our Gemini wrapper for LLM support
try:
    from src.core.gemini_client_wrapper import create_gemini_client_for_markitdown
    HAS_GEMINI_WRAPPER = True
except ImportError:
    HAS_GEMINI_WRAPPER = False

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MarkItDownParser(DocumentParser):
    """
    Parser implementation using MarkItDown for converting various file formats to Markdown.
    """
    
    def __init__(self):
        super().__init__()  # Initialize the base class (including _cancellation_flag)
        self.markdown_instance = None
        # Initialize MarkItDown instance
        if HAS_MARKITDOWN:
            try:
                # Initialize MarkItDown without LLM client for better performance
                # LLM client will only be used for image files when needed
                self.markdown_instance = MarkItDown()
                logger.info("MarkItDown initialized successfully")
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
        # Validate file first
        self.validate_file(file_path)
        
        # Check if MarkItDown is available
        if not HAS_MARKITDOWN or self.markdown_instance is None:
            raise ParserError("MarkItDown is not available. Please install with 'pip install markitdown[all]'")
        
        # Check for cancellation before starting
        if self._check_cancellation():
            raise DocumentProcessingError("Conversion cancelled")
        
        file_path_str = str(file_path)
        file_ext = Path(file_path).suffix.lower()
        
        try:
            # Run conversion in a separate thread to support cancellation
            result_container = {"result": None, "error": None, "completed": False}
            
            def conversion_worker():
                try:
                    # For image files, potentially use LLM if available
                    if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                        if HAS_GEMINI_WRAPPER:
                            try:
                                # Create Gemini-enabled instance for image processing
                                gemini_client = create_gemini_client_for_markitdown()
                                if gemini_client:
                                    llm_instance = MarkItDown(llm_client=gemini_client, llm_model="gemini-2.5-flash")
                                    result = llm_instance.convert(file_path_str)
                                else:
                                    # No Gemini client available, use standard conversion
                                    logger.info("Gemini client not available, using standard conversion for image")
                                    result = self.markdown_instance.convert(file_path_str)
                            except Exception as llm_error:
                                logger.warning(f"Gemini image processing failed, falling back to basic conversion: {llm_error}")
                                result = self.markdown_instance.convert(file_path_str)
                        else:
                            # No Gemini wrapper available, use standard conversion
                            logger.info("Gemini wrapper not available, using standard conversion for image")
                            result = self.markdown_instance.convert(file_path_str)
                    else:
                        # For non-image files, use standard conversion
                        result = self.markdown_instance.convert(file_path_str)
                    
                    result_container["result"] = result
                    result_container["completed"] = True
                except Exception as e:
                    result_container["error"] = e
                    result_container["completed"] = True
            
            # Start conversion in background thread
            conversion_thread = threading.Thread(target=conversion_worker, daemon=True)
            conversion_thread.start()
            
            # Wait for completion or cancellation
            while conversion_thread.is_alive():
                if self._check_cancellation():
                    logger.info("MarkItDown conversion cancelled by user")
                    # Give thread a moment to finish cleanly
                    conversion_thread.join(timeout=0.1)
                    raise DocumentProcessingError("Conversion cancelled")
                time.sleep(0.1)  # Check every 100ms
            
            # Ensure thread has completed
            conversion_thread.join()
            
            # Check for errors
            if result_container["error"]:
                raise result_container["error"]
            
            result = result_container["result"]
            if result is None:
                raise DocumentProcessingError("MarkItDown conversion returned no result")
            
            # Use the correct attribute - MarkItDown returns .text_content
            if hasattr(result, 'text_content') and result.text_content:
                return result.text_content
            elif hasattr(result, 'markdown') and result.markdown:
                return result.markdown
            elif hasattr(result, 'content') and result.content:
                return result.content
            else:
                # Fallback - convert result to string
                content = str(result)
                if content and content.strip():
                    return content
                else:
                    raise DocumentProcessingError("MarkItDown conversion returned empty content")
                
        except DocumentProcessingError:
            # Re-raise cancellation errors
            raise
        except Exception as e:
            logger.error(f"Error converting file with MarkItDown: {str(e)}")
            raise DocumentProcessingError(f"MarkItDown conversion failed: {str(e)}")
    
    @classmethod
    def get_name(cls) -> str:
        return "MarkItDown"
    
    @classmethod
    def get_supported_file_types(cls) -> Set[str]:
        """Return a set of supported file extensions."""
        return {".pdf", ".docx", ".xlsx", ".pptx", ".html", ".txt", ".md", ".json", ".xml", ".csv", ".jpg", ".jpeg", ".png"}
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if this parser is available."""
        return HAS_MARKITDOWN
    
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
        return "MarkItDown parser for converting various file formats to Markdown. Uses Gemini Flash 2.5 for advanced image analysis."


# Register the parser with the registry if available
if HAS_MARKITDOWN:
    ParserRegistry.register(MarkItDownParser)
    logger.info("MarkItDown parser registered successfully")
else:
    logger.warning("Could not register MarkItDown parser: Package not installed") 