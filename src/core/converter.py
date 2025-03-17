import tempfile
import logging
import time
import os
from pathlib import Path

# Use relative imports instead of absolute imports
from src.core.parser_factory import ParserFactory

# Import all parsers to ensure they're registered
import parsers

# Reference to the cancellation flag from ui.py
# This will be set by the UI when the cancel button is clicked
conversion_cancelled = None  # Will be a threading.Event object
# Flag to track if conversion is currently in progress
_conversion_in_progress = False

def set_cancellation_flag(flag):
    """Set the reference to the cancellation flag from ui.py"""
    global conversion_cancelled
    conversion_cancelled = flag

def is_conversion_in_progress():
    """Check if conversion is currently in progress"""
    global _conversion_in_progress
    return _conversion_in_progress

def check_cancellation():
    """Check if cancellation has been requested"""
    if conversion_cancelled and conversion_cancelled.is_set():
        logging.info("Cancellation detected in check_cancellation")
        return True
    return False

def safe_delete_file(file_path):
    """Safely delete a file with error handling"""
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception as e:
            logging.error(f"Error cleaning up temp file {file_path}: {e}")

def convert_file(file_path, parser_name, ocr_method_name, output_format):
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
    global conversion_cancelled, _conversion_in_progress
    
    # Set the conversion in progress flag
    _conversion_in_progress = True
    
    # Temporary file paths to clean up
    temp_input = None
    tmp_path = None
    
    # Ensure we clean up the flag when we're done
    try:
        if not file_path:
            return "Please upload a file.", None

        # Check for cancellation
        if check_cancellation():
            logging.info("Cancellation detected at start of convert_file")
            return "Conversion cancelled.", None

        # Create a temporary file with English filename
        try:
            original_ext = Path(file_path).suffix
            with tempfile.NamedTemporaryFile(suffix=original_ext, delete=False) as temp_file:
                temp_input = temp_file.name
                # Copy the content of original file to temp file
                with open(file_path, 'rb') as original:
                    # Read in smaller chunks and check for cancellation between chunks
                    chunk_size = 1024 * 1024  # 1MB chunks
                    while True:
                        # Check for cancellation frequently
                        if check_cancellation():
                            logging.info("Cancellation detected during file copy")
                            safe_delete_file(temp_input)
                            return "Conversion cancelled.", None
                        
                        chunk = original.read(chunk_size)
                        if not chunk:
                            break
                        temp_file.write(chunk)
            file_path = temp_input
        except Exception as e:
            safe_delete_file(temp_input)
            return f"Error creating temporary file: {e}", None

        # Check for cancellation again
        if check_cancellation():
            logging.info("Cancellation detected after file preparation")
            safe_delete_file(temp_input)
            return "Conversion cancelled.", None

        content = None
        try:
            # Use the parser factory to parse the document
            start = time.time()
            
            # Pass the cancellation flag to the parser factory
            content = ParserFactory.parse_document(
                file_path=file_path,
                parser_name=parser_name,
                ocr_method_name=ocr_method_name,
                output_format=output_format.lower(),
                cancellation_flag=conversion_cancelled  # Pass the flag to parsers
            )
            
            # If content indicates cancellation, return early
            if content == "Conversion cancelled.":
                logging.info("Parser reported cancellation")
                safe_delete_file(temp_input)
                return content, None
            
            duration = time.time() - start
            logging.info(f"Processed in {duration:.2f} seconds.")
            
            # Check for cancellation after processing
            if check_cancellation():
                logging.info("Cancellation detected after processing")
                safe_delete_file(temp_input)
                return "Conversion cancelled.", None
                
        except Exception as e:
            safe_delete_file(temp_input)
            return f"Error: {e}", None

        # Determine the file extension based on the output format
        if output_format == "Markdown":
            ext = ".md"
        elif output_format == "JSON":
            ext = ".json"
        elif output_format == "Text":
            ext = ".txt"
        elif output_format == "Document Tags":
            ext = ".doctags"
        else:
            ext = ".txt"

        # Check for cancellation again
        if check_cancellation():
            logging.info("Cancellation detected before output file creation")
            safe_delete_file(temp_input)
            return "Conversion cancelled.", None

        try:
            # Create a temporary file for download
            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False, encoding="utf-8") as tmp:
                tmp_path = tmp.name
                # Write in chunks and check for cancellation
                chunk_size = 10000  # characters
                for i in range(0, len(content), chunk_size):
                    # Check for cancellation
                    if check_cancellation():
                        logging.info("Cancellation detected during output file writing")
                        safe_delete_file(tmp_path)
                        safe_delete_file(temp_input)
                        return "Conversion cancelled.", None
                    
                    tmp.write(content[i:i+chunk_size])
            
            # Clean up the temporary input file
            safe_delete_file(temp_input)
            temp_input = None  # Mark as cleaned up
                
            return content, tmp_path
        except Exception as e:
            safe_delete_file(tmp_path)
            safe_delete_file(temp_input)
            return f"Error: {e}", None
    finally:
        # Always clean up any remaining temp files
        safe_delete_file(temp_input)
        if check_cancellation() and tmp_path:
            safe_delete_file(tmp_path)
            
        # Always clear the conversion in progress flag when done
        _conversion_in_progress = False
