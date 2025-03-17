from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import pypdfium2 as pdfium

from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend


class PyPdfiumParser(DocumentParser):
    """Parser implementation using PyPdfium."""
    
    @classmethod
    def get_name(cls) -> str:
        return "PyPdfium"
    
    @classmethod
    def get_supported_ocr_methods(cls) -> List[Dict[str, Any]]:
        return [
            {
                "id": "no_ocr",
                "name": "No OCR",
                "default_params": {}
            },
            {
                "id": "easyocr",
                "name": "EasyOCR",
                "default_params": {"languages": ["en"]}
            }
        ]
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """Parse a document using PyPdfium."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # Configure OCR based on the method
        if ocr_method == "easyocr":
            pipeline_options.do_ocr = True
            # Apply any custom parameters from kwargs
            if "languages" in kwargs:
                pipeline_options.ocr_options.lang = kwargs["languages"]
        else:
            pipeline_options.do_ocr = False
        
        # Create the converter
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        
        # Convert the document
        result = converter.convert(Path(file_path))
        doc = result.document
        
        # Return the content in the requested format
        output_format = kwargs.get("output_format", "markdown")
        if output_format.lower() == "json":
            return json.dumps(doc.export_to_dict(), ensure_ascii=False, indent=2)
        elif output_format.lower() == "text":
            return doc.export_to_text()
        elif output_format.lower() == "document_tags":
            return doc.export_to_document_tokens()
        else:
            return doc.export_to_markdown()


# Register the parser with the registry
ParserRegistry.register(PyPdfiumParser) 