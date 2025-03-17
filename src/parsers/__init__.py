"""Parser implementations for document conversion."""

# Import all parsers to ensure they're registered
from src.parsers.docling_parser import DoclingParser
from src.parsers.marker_parser import MarkerParser
from src.parsers.pypdfium_parser import PyPdfiumParser
from src.parsers.gemini_flash_parser import GeminiFlashParser
from src.parsers.got_ocr_parser import GotOcrParser

# You can add new parsers here in the future 

# This file makes the parsers directory a Python package 