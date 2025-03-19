import logging
from typing import Optional, Dict, Any
import os
from pathlib import Path

# Import the LaTeX converter utility
from src.utils.latex_converter import convert_latex_to_markdown

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_got_ocr_output(output_text: str, ocr_method: str, output_format: str) -> str:
    """
    Process the output from GOT-OCR parser and convert if needed.
    
    Args:
        output_text: The raw output text from the GOT-OCR parser
        ocr_method: The OCR method used (Plain Text, Formatted Text)
        output_format: The desired output format (Markdown, etc.)
        
    Returns:
        str: The processed text
    """
    if not output_text:
        return ""
        
    # If not using formatted text or not requesting Markdown, return the original text
    if ocr_method.lower() != "formatted text" or output_format.lower() != "markdown":
        return output_text
        
    # Process the formatted text (LaTeX) into enhanced Markdown
    logger.info("Converting LaTeX output to enhanced Markdown format")
    try:
        markdown_text = convert_latex_to_markdown(output_text)
        logger.info("LaTeX to Markdown conversion successful")
        return markdown_text
    except Exception as e:
        logger.error(f"Error converting LaTeX to Markdown: {str(e)}")
        # Return the original text if conversion fails
        return output_text 