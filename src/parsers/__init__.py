"""Parser implementations for document conversion."""

# Import all parsers to ensure they're registered
from src.parsers.gemini_flash_parser import GeminiFlashParser
from src.parsers.got_ocr_parser import GotOcrParser
from src.parsers.mistral_ocr_parser import MistralOcrParser

# Import Docling parser if available
try:
    from src.parsers.docling_parser import DoclingParser
    print("Docling parser imported successfully")
except ImportError as e:
    print(f"Error importing Docling parser: {str(e)}")

# Import MarkItDown parser if available - needs to be imported last so it's default
try:
    from src.parsers.markitdown_parser import MarkItDownParser
    print("MarkItDown parser imported successfully")
except ImportError as e:
    print(f"Error importing MarkItDown parser: {str(e)}")

# You can add new parsers here in the future 

# This file makes the parsers directory a Python package 