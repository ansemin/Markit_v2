"""
Document processing service layer.
"""
import tempfile
import logging
import time
import os
import threading
from pathlib import Path
from typing import Optional, Tuple, Any

from src.core.config import config
from src.core.exceptions import (
    DocumentProcessingError, 
    FileSizeLimitError, 
    UnsupportedFileTypeError,
    ConversionError
)
from src.core.parser_factory import ParserFactory
from src.core.latex_to_markdown_converter import convert_latex_to_markdown


class DocumentService:
    """Service for handling document processing operations."""
    
    def __init__(self):
        self._conversion_in_progress = False
        self._cancellation_flag: Optional[threading.Event] = None
    
    def set_cancellation_flag(self, flag: threading.Event) -> None:
        """Set the cancellation flag for this service."""
        self._cancellation_flag = flag
    
    def is_conversion_in_progress(self) -> bool:
        """Check if conversion is currently in progress."""
        return self._conversion_in_progress
    
    def _check_cancellation(self) -> bool:
        """Check if cancellation has been requested."""
        if self._cancellation_flag and self._cancellation_flag.is_set():
            logging.info("Cancellation detected in document service")
            return True
        return False
    
    def _safe_delete_file(self, file_path: Optional[str]) -> None:
        """Safely delete a file with error handling."""
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception as e:
                logging.error(f"Error cleaning up temp file {file_path}: {e}")
    
    def _validate_file(self, file_path: str) -> None:
        """Validate file size and type."""
        if not os.path.exists(file_path):
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > config.app.max_file_size:
            raise FileSizeLimitError(
                f"File size ({file_size} bytes) exceeds maximum allowed size "
                f"({config.app.max_file_size} bytes)"
            )
        
        # Check file extension
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in config.app.allowed_extensions:
            raise UnsupportedFileTypeError(
                f"File type '{file_ext}' is not supported. "
                f"Allowed types: {', '.join(config.app.allowed_extensions)}"
            )
    
    def _create_temp_file(self, original_path: str) -> str:
        """Create a temporary file with English filename."""
        original_ext = Path(original_path).suffix
        
        with tempfile.NamedTemporaryFile(suffix=original_ext, delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Copy content in chunks with cancellation checks
            with open(original_path, 'rb') as original:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    if self._check_cancellation():
                        self._safe_delete_file(temp_path)
                        raise ConversionError("Conversion cancelled during file copy")
                    
                    chunk = original.read(chunk_size)
                    if not chunk:
                        break
                    temp_file.write(chunk)
        
        return temp_path
    
    def _process_latex_content(self, content: str, parser_name: str, ocr_method_name: str) -> str:
        """Process LaTeX content for GOT-OCR formatted text."""
        if (parser_name == "GOT-OCR (jpg,png only)" and 
            ocr_method_name == "Formatted Text" and 
            config.api.google_api_key):
            
            logging.info("Converting LaTeX output to Markdown using Gemini API")
            start_convert = time.time()
            
            if self._check_cancellation():
                raise ConversionError("Conversion cancelled before LaTeX conversion")
            
            try:
                markdown_content = convert_latex_to_markdown(content)
                if markdown_content:
                    logging.info(f"LaTeX conversion completed in {time.time() - start_convert:.2f} seconds")
                    return markdown_content
                else:
                    logging.warning("LaTeX to Markdown conversion failed, using raw LaTeX output")
            except Exception as e:
                logging.error(f"Error converting LaTeX to Markdown: {str(e)}")
                # Continue with original content on error
        
        return content
    
    def _create_output_file(self, content: str, output_format: str, original_file_path: Optional[str] = None) -> str:
        """Create output file with proper extension and preserved filename."""
        # Determine file extension
        format_extensions = {
            "markdown": ".md",
            "json": ".json", 
            "text": ".txt",
            "document tags": ".doctags"
        }
        ext = format_extensions.get(output_format.lower(), ".txt")
        
        if self._check_cancellation():
            raise ConversionError("Conversion cancelled before output file creation")
        
        # Create output filename based on original filename if provided
        if original_file_path:
            original_name = Path(original_file_path).stem  # Get filename without extension
            # Clean the filename to be filesystem-safe while preserving spaces and common characters
            clean_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_', '.', '(', ')')).strip()
            # Replace multiple spaces with single spaces
            clean_name = ' '.join(clean_name.split())
            if not clean_name:  # Fallback if cleaning removes everything
                clean_name = "converted_document"
            
            # Create output file in temp directory with proper name
            temp_dir = tempfile.gettempdir()
            output_filename = f"{clean_name}{ext}"
            tmp_path = os.path.join(temp_dir, output_filename)
            
            # Handle filename conflicts by adding a number suffix
            counter = 1
            base_path = tmp_path
            while os.path.exists(tmp_path):
                name_part = f"{clean_name}_{counter}"
                tmp_path = os.path.join(temp_dir, f"{name_part}{ext}")
                counter += 1
        else:
            # Fallback to random temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False, encoding="utf-8") as tmp:
                tmp_path = tmp.name
        
        # Write content to file
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                # Write in chunks with cancellation checks
                chunk_size = 10000  # characters
                for i in range(0, len(content), chunk_size):
                    if self._check_cancellation():
                        self._safe_delete_file(tmp_path)
                        raise ConversionError("Conversion cancelled during output file writing")
                    
                    f.write(content[i:i+chunk_size])
        except Exception as e:
            self._safe_delete_file(tmp_path)
            raise ConversionError(f"Failed to write output file: {str(e)}")
        
        return tmp_path
    
    def convert_document(
        self, 
        file_path: str, 
        parser_name: str, 
        ocr_method_name: str, 
        output_format: str
    ) -> Tuple[str, Optional[str]]:
        """
        Convert a document using the specified parser and OCR method.
        
        Args:
            file_path: Path to the input file
            parser_name: Name of the parser to use
            ocr_method_name: Name of the OCR method to use
            output_format: Output format (Markdown, JSON, Text, Document Tags)
            
        Returns:
            Tuple of (content, output_file_path)
            
        Raises:
            DocumentProcessingError: For general processing errors
            FileSizeLimitError: When file is too large
            UnsupportedFileTypeError: For unsupported file types
            ConversionError: When conversion fails or is cancelled
        """
        if not file_path:
            raise DocumentProcessingError("No file provided")
        
        self._conversion_in_progress = True
        temp_input = None
        output_path = None
        
        try:
            # Validate input file
            self._validate_file(file_path)
            
            if self._check_cancellation():
                raise ConversionError("Conversion cancelled")
            
            # Create temporary file with English name
            temp_input = self._create_temp_file(file_path)
            
            if self._check_cancellation():
                raise ConversionError("Conversion cancelled")
            
            # Process document using parser factory
            start_time = time.time()
            content = ParserFactory.parse_document(
                file_path=temp_input,
                parser_name=parser_name,
                ocr_method_name=ocr_method_name,
                output_format=output_format.lower(),
                cancellation_flag=self._cancellation_flag
            )
            
            if content == "Conversion cancelled.":
                raise ConversionError("Conversion cancelled by parser")
            
            duration = time.time() - start_time
            logging.info(f"Document processed in {duration:.2f} seconds")
            
            if self._check_cancellation():
                raise ConversionError("Conversion cancelled")
            
            # Process LaTeX content if needed
            content = self._process_latex_content(content, parser_name, ocr_method_name)
            
            if self._check_cancellation():
                raise ConversionError("Conversion cancelled")
            
            # Create output file
            output_path = self._create_output_file(content, output_format, file_path)
            
            return content, output_path
            
        except (DocumentProcessingError, FileSizeLimitError, UnsupportedFileTypeError, ConversionError):
            # Re-raise our custom exceptions
            self._safe_delete_file(temp_input)
            self._safe_delete_file(output_path)
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            self._safe_delete_file(temp_input)
            self._safe_delete_file(output_path)
            raise DocumentProcessingError(f"Unexpected error during conversion: {str(e)}")
        finally:
            # Clean up temp input file
            self._safe_delete_file(temp_input)
            
            # Clean up output file if cancelled
            if self._check_cancellation() and output_path:
                self._safe_delete_file(output_path)
            
            self._conversion_in_progress = False