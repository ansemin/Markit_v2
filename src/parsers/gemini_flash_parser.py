from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import os
import json
import tempfile
import base64
from PIL import Image
import io

from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry

# Import the Google Gemini API client
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load API key from environment variable
api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key is available and print a message if not
if not api_key:
    print("Warning: GOOGLE_API_KEY environment variable not found. Gemini Flash parser may not work.")

class GeminiFlashParser(DocumentParser):
    """Parser that uses Google's Gemini Flash 2.0 to convert documents to markdown."""

    @classmethod
    def get_name(cls) -> str:
        return "Gemini Flash"

    @classmethod
    def get_supported_ocr_methods(cls) -> List[Dict[str, Any]]:
        return [
            {
                "id": "none",
                "name": "None",
                "default_params": {}
            }
        ]
    
    @classmethod
    def get_description(cls) -> str:
        return "Gemini Flash 2.0 parser for converting documents and images to markdown"
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """Parse a document using Gemini Flash 2.0."""
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "The Google Gemini API client is not installed. "
                "Please install it with 'pip install google-genai'."
            )
        
        # Use the globally loaded API key
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Please set it to your Gemini API key."
            )
        
        try:
            # Configure the Gemini API with the API key
            genai.configure(api_key=api_key)
            
            # Determine file type based on extension
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            # Read the file content
            file_content = file_path.read_bytes()
            
            # Determine MIME type based on file extension
            mime_type = self._get_mime_type(file_extension)
            
            # Create a multipart content with the file
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Set up the prompt
            prompt = """
            Convert this document to markdown format. 
            Preserve the structure, headings, lists, tables, and formatting as much as possible.
            For images, include a brief description in markdown image syntax.
            """
            
            # Generate the response
            response = model.generate_content(
                contents=[
                    prompt,
                    {
                        "mime_type": mime_type,
                        "data": file_content
                    }
                ],
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            
            # Extract the markdown text from the response
            markdown_text = response.text
            
            return markdown_text
            
        except Exception as e:
            error_message = f"Error parsing document with Gemini Flash: {str(e)}"
            print(error_message)
            return f"# Error\n\n{error_message}\n\nPlease check your API key and try again."
    
    def _get_mime_type(self, file_extension: str) -> str:
        """Get the MIME type for a file extension."""
        mime_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".html": "text/html",
            ".htm": "text/html",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }
        
        return mime_types.get(file_extension, "application/octet-stream")


# Register the parser with the registry
if GEMINI_AVAILABLE:
    ParserRegistry.register(GeminiFlashParser)
else:
    print("Gemini Flash parser not registered: google-genai package not installed") 