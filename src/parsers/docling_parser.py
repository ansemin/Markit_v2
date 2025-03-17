from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import os
import shutil

from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.models.tesseract_ocr_model import TesseractOcrOptions
from docling.models.tesseract_ocr_cli_model import TesseractCliOcrOptions
from docling.models.ocr_mac_model import OcrMacOptions


class DoclingParser(DocumentParser):
    """Parser implementation using Docling."""
    
    @classmethod
    def get_name(cls) -> str:
        return "Docling"
    
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
            },
            {
                "id": "easyocr_cpu",
                "name": "EasyOCR (CPU only)",
                "default_params": {"languages": ["en"], "use_gpu": False}
            },
            {
                "id": "tesseract",
                "name": "Tesseract",
                "default_params": {}
            },
            {
                "id": "tesseract_cli",
                "name": "Tesseract CLI",
                "default_params": {}
            },
            {
                "id": "ocrmac",
                "name": "ocrmac",
                "default_params": {}
            },
            {
                "id": "full_force_ocr",
                "name": "Full Force OCR",
                "default_params": {}
            }
        ]
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """Parse a document using Docling."""
        # Special case for full force OCR
        if ocr_method == "full_force_ocr":
            return self._apply_full_force_ocr(file_path)
        
        # Regular Docling parsing
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # Configure OCR based on the method
        if ocr_method == "no_ocr":
            pipeline_options.do_ocr = False
        elif ocr_method == "easyocr":
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options.lang = kwargs.get("languages", ["en"])
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4, device=AcceleratorDevice.AUTO
            )
        elif ocr_method == "easyocr_cpu":
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options.lang = kwargs.get("languages", ["en"])
            pipeline_options.ocr_options.use_gpu = False
        elif ocr_method == "tesseract":
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = TesseractOcrOptions()
        elif ocr_method == "tesseract_cli":
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = TesseractCliOcrOptions()
        elif ocr_method == "ocrmac":
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = OcrMacOptions()
        
        # Create the converter
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
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
    
    def _apply_full_force_ocr(self, file_path: Union[str, Path]) -> str:
        """Apply full force OCR to a document."""
        input_doc = Path(file_path)
        file_extension = input_doc.suffix.lower()
        
        # Debug information
        print(f"Applying full force OCR to file: {input_doc} (type: {file_extension})")
        
        # Basic pipeline setup
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # Find tesseract executable
        tesseract_path = shutil.which("tesseract") or "/usr/bin/tesseract"
        print(f"Using tesseract at: {tesseract_path}")
        
        # Configure OCR options
        ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)  # Using standard options instead of CLI
        pipeline_options.ocr_options = ocr_options
        
        # Set up format options based on file type
        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
        
        # Handle image files
        if file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']:
            print(f"Processing as image file: {file_extension}")
            format_options[InputFormat.IMAGE] = PdfFormatOption(pipeline_options=pipeline_options)
        
        # Try full force OCR with standard options
        try:
            converter = DocumentConverter(format_options=format_options)
            result = converter.convert(input_doc)
            return result.document.export_to_markdown()
        except Exception as e:
            print(f"Error with standard OCR: {e}")
            print(f"Attempting fallback to tesseract_cli OCR...")
            return self.parse(file_path, ocr_method="tesseract_cli")


# Register the parser with the registry
ParserRegistry.register(DoclingParser) 