from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import subprocess
import tempfile
import os
import json

from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


class MarkerParser(DocumentParser):
    """Parser implementation using Marker."""
    
    @classmethod
    def get_name(cls) -> str:
        return "Marker"
    
    @classmethod
    def get_supported_ocr_methods(cls) -> List[Dict[str, Any]]:
        return [
            {
                "id": "no_ocr",
                "name": "No OCR",
                "default_params": {}
            },
            {
                "id": "force_ocr",
                "name": "Force OCR",
                "default_params": {}
            }
        ]
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """Parse a document using Marker."""
        force_ocr = ocr_method == "force_ocr"
        
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config={"force_ocr": force_ocr}
        )
        rendered = converter(str(file_path))
        content, _, _ = text_from_rendered(rendered)
        
        # Format the content based on the requested output format
        output_format = kwargs.get("output_format", "markdown")
        if output_format.lower() == "json":
            return json.dumps({"content": content}, ensure_ascii=False, indent=2)
        elif output_format.lower() == "text":
            return content.replace("#", "").replace("*", "").replace("_", "")
        elif output_format.lower() == "document_tags":
            return f"<doc>\n{content}\n</doc>"
        else:
            return content


# Register the parser with the registry
ParserRegistry.register(MarkerParser) 