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
from src.core.config import config

# Import the Google Gemini API client
try:
    from google import genai
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
            # Determine file type based on extension
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            # Read the file content
            file_content = file_path.read_bytes()
            
            # Determine MIME type based on file extension
            mime_type = self._get_mime_type(file_extension)
            
            # Create a client and use the model
            client = genai.Client(api_key=api_key)
            
            # Set up the prompt
            prompt = """
            Convert this document to markdown format. 
            Preserve the structure, headings, lists, tables, and formatting as much as possible.
            For images, include a brief description in markdown image syntax.
            Return only the markdown content, no other text.
            """
            
            # Generate the response
            response = client.models.generate_content(
                model=config.model.gemini_model,
                contents=[
                    prompt,
                    genai.types.Part.from_bytes(
                        data=file_content,
                        mime_type=mime_type
                    )
                ],
                config={
                    "temperature": config.model.temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": config.model.max_tokens,
                }
            )
            
            # Extract the markdown text from the response
            markdown_text = response.text
            
            return markdown_text
            
        except Exception as e:
            error_message = f"Error parsing document with Gemini Flash: {str(e)}"
            print(error_message)
            return f"# Error\n\n{error_message}\n\nPlease check your API key and try again."
    
    def parse_multiple(self, file_paths: List[Union[str, Path]], processing_type: str = "combined", original_filenames: Optional[List[str]] = None, **kwargs) -> str:
        """Parse multiple documents using Gemini Flash 2.0."""
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "The Google Gemini API client is not installed. "
                "Please install it with 'pip install google-genai'."
            )
        
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Please set it to your Gemini API key."
            )
        
        try:
            # Convert to Path objects and validate
            path_objects = [Path(fp) for fp in file_paths]
            self._validate_batch_files(path_objects)
            
            # Check for cancellation
            if self._check_cancellation():
                return "Conversion cancelled."
            
            # Create client
            client = genai.Client(api_key=api_key)
            
            # Create contents for API call
            contents = self._create_batch_contents(path_objects, processing_type, original_filenames)
            
            # Check for cancellation before API call
            if self._check_cancellation():
                return "Conversion cancelled."
            
            # Generate the response
            response = client.models.generate_content(
                model=config.model.gemini_model,
                contents=contents,
                config={
                    "temperature": config.model.temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": config.model.max_tokens,
                }
            )
            
            # Format the output based on processing type
            formatted_output = self._format_batch_output(response.text, path_objects, processing_type, original_filenames)
            
            return formatted_output
            
        except Exception as e:
            error_message = f"Error parsing multiple documents with Gemini Flash: {str(e)}"
            print(error_message)
            return f"# Error\n\n{error_message}\n\nPlease check your API key and try again."
    
    def _validate_batch_files(self, file_paths: List[Path]) -> None:
        """Validate batch of files for multi-document processing."""
        # Check file count limit
        if len(file_paths) == 0:
            raise ValueError("No files provided for processing")
        if len(file_paths) > 5:
            raise ValueError("Maximum 5 files allowed for batch processing")
        
        # Check individual files and calculate total size
        total_size = 0
        for file_path in file_paths:
            if not file_path.exists():
                raise ValueError(f"File not found: {file_path}")
            
            file_size = file_path.stat().st_size
            total_size += file_size
            
            # Check individual file size (reasonable limit per file)
            if file_size > 10 * 1024 * 1024:  # 10MB per file
                raise ValueError(f"Individual file size exceeds 10MB: {file_path.name}")
        
        # Check combined size limit
        if total_size > 20 * 1024 * 1024:  # 20MB total
            raise ValueError(f"Combined file size ({total_size / (1024*1024):.1f}MB) exceeds 20MB limit")
        
        # Validate file types
        for file_path in file_paths:
            file_extension = file_path.suffix.lower()
            mime_type = self._get_mime_type(file_extension)
            if mime_type == "application/octet-stream":
                raise ValueError(f"Unsupported file type: {file_path.name}. Gemini supports: PDF, TXT, HTML, CSS, MD, CSV, XML, RTF, JS, PY, and image files.")
            # Check if it's a supported MIME type for Gemini
            if mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                           "application/msword", 
                           "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                           "application/vnd.ms-powerpoint",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           "application/vnd.ms-excel"]:
                raise ValueError(f"File type not supported by Gemini: {file_path.name}. Gemini supports: PDF, TXT, HTML, CSS, MD, CSV, XML, RTF, JS, PY, and image files.")
    
    def _create_batch_contents(self, file_paths: List[Path], processing_type: str, original_filenames: Optional[List[str]] = None) -> List[Any]:
        """Create contents list for batch API call."""
        # Create the prompt based on processing type
        prompt = self._create_batch_prompt(file_paths, processing_type, original_filenames)
        
        # Start with the prompt
        contents = [prompt]
        
        # Add each file as a content part
        for file_path in file_paths:
            file_content = file_path.read_bytes()
            mime_type = self._get_mime_type(file_path.suffix.lower())
            
            contents.append(
                genai.types.Part.from_bytes(
                    data=file_content,
                    mime_type=mime_type
                )
            )
        
        return contents
    
    def _create_batch_prompt(self, file_paths: List[Path], processing_type: str, original_filenames: Optional[List[str]] = None) -> str:
        """Create appropriate prompt for batch processing."""
        # Use original filenames if provided, otherwise use temp file names
        if original_filenames:
            file_names = original_filenames
        else:
            file_names = [fp.name for fp in file_paths]
        file_list = "\n".join([f"- {name}" for name in file_names])
        
        base_prompt = f"""I will provide you with {len(file_paths)} documents to process:
{file_list}

"""
        
        if processing_type == "combined":
            return base_prompt + """Please convert all documents to a single, cohesive markdown document. 
Merge the content logically, remove duplicate information, and create a unified structure with clear headings.
Preserve important formatting, tables, lists, and structure from all documents.
For images, include brief descriptions in markdown image syntax.
Return only the combined markdown content, no other text."""
            
        elif processing_type == "individual":
            return base_prompt + """Please convert each document to markdown format and present them as separate sections.
For each document, create a clear section header with the document name.
Preserve the structure, headings, lists, tables, and formatting within each section.
For images, include brief descriptions in markdown image syntax.
Return the content in this format:

# Document 1: [filename]
[converted content]

# Document 2: [filename]  
[converted content]

Return only the markdown content, no other text."""
            
        elif processing_type == "summary":
            return base_prompt + """Please create a comprehensive analysis with two parts:

1. EXECUTIVE SUMMARY: A concise overview summarizing the key points from all documents
2. DETAILED SECTIONS: Individual converted sections for each document

Structure the output as:

# Executive Summary
[Brief summary of key findings and themes across all documents]

# Detailed Analysis

## Document 1: [filename]
[converted content]

## Document 2: [filename]
[converted content]

Preserve formatting, tables, lists, and structure throughout.
For images, include brief descriptions in markdown image syntax.
Return only the markdown content, no other text."""
            
        elif processing_type == "comparison":
            return base_prompt + """Please create a comparative analysis of these documents:

1. Create a comparison table highlighting key differences and similarities
2. Provide individual document summaries
3. Include a section on cross-document insights

Structure the output as:

# Document Comparison Analysis

## Comparison Table
| Aspect | Document 1 | Document 2 | Document 3 | ... |
|--------|------------|------------|------------|-----|
| [Key aspects found across documents] | | | | |

## Individual Document Summaries

### Document 1: [filename]
[Key points and content summary]

### Document 2: [filename]
[Key points and content summary]

## Cross-Document Insights
[Analysis of patterns, contradictions, or complementary information across documents]

Preserve important formatting and structure.
For images, include brief descriptions in markdown image syntax.
Return only the markdown content, no other text."""
        
        else:
            # Fallback to combined
            return self._create_batch_prompt(file_paths, "combined")
    
    def _format_batch_output(self, response_text: str, file_paths: List[Path], processing_type: str, original_filenames: Optional[List[str]] = None) -> str:
        """Format the batch processing output."""
        # Add metadata header using original filenames if provided
        if original_filenames:
            file_names = original_filenames
        else:
            file_names = [fp.name for fp in file_paths]
        
        header = f"""<!-- Multi-Document Processing Results -->
<!-- Processing Type: {processing_type} -->
<!-- Files Processed: {len(file_paths)} -->
<!-- File Names: {', '.join(file_names)} -->

"""
        
        return header + response_text
    
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
            ".csv": "text/csv",
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