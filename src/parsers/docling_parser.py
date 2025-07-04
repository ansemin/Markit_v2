# Import spaces module for ZeroGPU support - Must be first import
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
import tempfile

# Force CPU-only mode for EasyOCR and other CUDA libraries
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['USE_TORCH'] = '1'
os.environ['EASYOCR_GPU'] = 'False'

# Import the parser interface and registry
from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry
from src.core.exceptions import DocumentProcessingError, ParserError
from src.core.config import config

# Check for Docling availability
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TesseractOcrOptions
    from docling.document_converter import PdfFormatOption
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False
    logging.warning("Docling package not installed. Please install with 'pip install docling'")

# Gemini availability
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DoclingParser(DocumentParser):
    """
    Parser implementation using Docling for converting documents to Markdown.
    Supports advanced PDF understanding, OCR, and multiple document formats.
    """
    
    def __init__(self):
        super().__init__()  # Initialize the base class (including _cancellation_flag)
        self.converter = None
        self.gpu_converter = None
        
        # Don't initialize converters here to avoid CUDA issues
        # They will be created on-demand in the parse methods
        logger.info("Docling parser initialized (converters will be created on-demand)")
    
    def _create_converter_with_options(self, ocr_method: str, **kwargs) -> DocumentConverter:
        """Create a DocumentConverter with specific OCR options."""
        pipeline_options = PdfPipelineOptions()
        
        # Enable OCR by default
        pipeline_options.do_ocr = True
        
        # Configure OCR method
        if ocr_method == "docling_tesseract":
            pipeline_options.ocr_options = TesseractOcrOptions()
        elif ocr_method == "docling_easyocr":
            pipeline_options.ocr_options = EasyOcrOptions()
        else:  # Default to EasyOCR
            pipeline_options.ocr_options = EasyOcrOptions()
        
        # Configure advanced features
        pipeline_options.do_table_structure = kwargs.get('enable_tables', True)
        pipeline_options.do_code_enrichment = kwargs.get('enable_code_enrichment', False)
        pipeline_options.do_formula_enrichment = kwargs.get('enable_formula_enrichment', False)
        pipeline_options.do_picture_classification = kwargs.get('enable_picture_classification', False)
        pipeline_options.generate_picture_images = kwargs.get('generate_picture_images', False)
        
        # Create converter with options
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        return converter
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """
        Parse a document and return its content as Markdown.
        
        Args:
            file_path: Path to the document
            ocr_method: OCR method to use ('docling_default', 'docling_tesseract', 'docling_easyocr')
            **kwargs: Additional options for Docling processing
        
        Returns:
            str: Markdown representation of the document
        """
        # Validate file first
        self.validate_file(file_path)
        
        # Check if Docling is available
        if not HAS_DOCLING:
            raise ParserError("Docling is not available. Please install with 'pip install docling'")
        
        # Check for cancellation before starting
        if self._check_cancellation():
            raise DocumentProcessingError("Conversion cancelled")
        
        try:
            # Try ZeroGPU first, fallback to CPU
            if HAS_SPACES:
                try:
                    logger.info("Attempting Docling processing with ZeroGPU")
                    # Filter kwargs to avoid pickle issues
                    safe_kwargs = {}
                    for key, value in kwargs.items():
                        if not key.startswith('_') and not hasattr(value, '__call__'):
                            try:
                                import pickle
                                pickle.dumps(value)
                                safe_kwargs[key] = value
                            except (TypeError, pickle.PickleError):
                                logger.debug(f"Skipping unpicklable kwarg: {key}")
                    
                    result = self._process_with_gpu(str(file_path), ocr_method, **safe_kwargs)
                    return result
                except Exception as e:
                    if "pickle" in str(e).lower():
                        logger.warning(f"ZeroGPU pickle error: {str(e)}")
                    elif "cuda" in str(e).lower():
                        logger.warning(f"ZeroGPU CUDA error: {str(e)}")
                    else:
                        logger.warning(f"ZeroGPU processing failed: {str(e)}")
                    logger.info("Falling back to CPU processing")
            
            # Fallback to CPU processing
            result = self._process_with_cpu(str(file_path), ocr_method, **kwargs)
            return result
            
        except Exception as e:
            logger.error(f"Error converting file with Docling: {str(e)}")
            raise DocumentProcessingError(f"Docling conversion failed: {str(e)}")
    
    def _process_with_cpu(self, file_path: str, ocr_method: Optional[str] = None, **kwargs) -> str:
        """Process document with CPU-only Docling converter."""
        logger.info("Processing with CPU-only Docling converter")
        
        # Create CPU converter if not exists
        if self.converter is None:
            self.converter = self._create_cpu_converter(ocr_method, **kwargs)
        
        # Convert the document
        result = self.converter.convert(file_path)
        
        # Check for cancellation after processing
        if self._check_cancellation():
            raise DocumentProcessingError("Conversion cancelled")
        
        # Export to markdown
        return result.document.export_to_markdown()
    
    def _create_cpu_converter(self, ocr_method: Optional[str] = None, **kwargs) -> DocumentConverter:
        """Create a CPU-only DocumentConverter with proper OCR fallback."""
        # Configure CPU-only accelerator
        accelerator_options = AcceleratorOptions(
            num_threads=4, 
            device=AcceleratorDevice.CPU
        )
        
        # Create pipeline options with CPU-only accelerator
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # Configure OCR method - use EasyOCR with CPU enforcement
        pipeline_options.ocr_options = EasyOcrOptions()
        logger.info("Using EasyOCR (CPU-only)")
        
        # Configure advanced features
        pipeline_options.do_table_structure = kwargs.get('enable_tables', True)
        pipeline_options.do_code_enrichment = kwargs.get('enable_code_enrichment', False)
        pipeline_options.do_formula_enrichment = kwargs.get('enable_formula_enrichment', False)
        pipeline_options.do_picture_classification = kwargs.get('enable_picture_classification', False)
        pipeline_options.generate_picture_images = kwargs.get('generate_picture_images', False)
        
        # Create converter with CPU-only configuration
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
    
    # Define the GPU-decorated function for ZeroGPU
    if HAS_SPACES:
        @spaces.GPU(duration=120)  # Allocate GPU for up to 2 minutes
        def _process_with_gpu(self, file_path: str, ocr_method: Optional[str] = None, **kwargs) -> str:
            """Process document with GPU-accelerated Docling converter.
            
            IMPORTANT: All model loading and CUDA operations must happen inside this method.
            """
            logger.info("Processing with ZeroGPU allocation")
            
            # Configure GPU accelerator
            accelerator_options = AcceleratorOptions(
                num_threads=4, 
                device=AcceleratorDevice.CUDA
            )
            
            # Create pipeline options with GPU accelerator
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = accelerator_options
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            
            # Configure OCR method - use EasyOCR
            pipeline_options.ocr_options = EasyOcrOptions()
            
            # Configure advanced features
            pipeline_options.do_table_structure = kwargs.get('enable_tables', True)
            pipeline_options.do_code_enrichment = kwargs.get('enable_code_enrichment', False)
            pipeline_options.do_formula_enrichment = kwargs.get('enable_formula_enrichment', False)
            pipeline_options.do_picture_classification = kwargs.get('enable_picture_classification', False)
            pipeline_options.generate_picture_images = kwargs.get('generate_picture_images', False)
            
            # Create converter with GPU configuration inside the decorated function
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )
            
            # Convert the document
            result = converter.convert(file_path)
            
            # Export to markdown
            markdown_content = result.document.export_to_markdown()
            
            # Clean up to free memory
            del converter
            import gc
            gc.collect()
            
            return markdown_content
    else:
        # Define a dummy method if spaces is not available
        def _process_with_gpu(self, file_path: str, ocr_method: Optional[str] = None, **kwargs) -> str:
            # This should never be called if HAS_SPACES is False
            return self._process_with_cpu(file_path, ocr_method, **kwargs)
    
    @classmethod
    def get_name(cls) -> str:
        return "Docling"
    
    @classmethod
    def get_supported_file_types(cls) -> Set[str]:
        """Return a set of supported file extensions."""
        return {
            # PDF files
            ".pdf",
            # Image files
            ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
            # Office documents
            ".docx", ".xlsx", ".pptx",
            # Web and markup
            ".html", ".xhtml", ".md",
            # Other formats
            ".csv"
        }
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if this parser is available."""
        return HAS_DOCLING
    
    @classmethod
    def get_supported_ocr_methods(cls) -> List[Dict[str, Any]]:
        """Return list of supported OCR methods."""
        return [
            {
                "id": "docling_default",
                "name": "EasyOCR",
                "default_params": {
                    "enable_tables": True,
                    "enable_code_enrichment": False,
                    "enable_formula_enrichment": False,
                    "enable_picture_classification": False,
                    "generate_picture_images": False
                }
            }
        ]
    
    @classmethod
    def get_description(cls) -> str:
        return "Docling parser with advanced PDF understanding, table structure recognition, and multiple OCR engines"

    def _validate_batch_files(self, file_paths: List[Path]) -> None:
        """Validate batch of files (size, count, type) for multi-document processing."""
        if len(file_paths) == 0:
            raise DocumentProcessingError("No files provided for processing")
        if len(file_paths) > 5:
            raise DocumentProcessingError("Maximum 5 files allowed for batch processing")

        total_size = 0
        for fp in file_paths:
            if not fp.exists():
                raise DocumentProcessingError(f"File not found: {fp}")
            size = fp.stat().st_size
            if size > 10 * 1024 * 1024:  # 10 MB
                raise DocumentProcessingError(f"Individual file size exceeds 10MB: {fp.name}")
            total_size += size
        if total_size > 20 * 1024 * 1024:
            raise DocumentProcessingError(f"Combined file size ({total_size / (1024*1024):.1f}MB) exceeds 20MB limit")

    def _create_batch_prompt(self, file_paths: List[Path], processing_type: str, original_filenames: Optional[List[str]] = None) -> str:
        """Create a natural-language prompt for Gemini post-processing."""
        names = original_filenames if original_filenames else [p.name for p in file_paths]
        file_list = "\n".join(f"- {n}" for n in names)
        base = f"I will provide you with {len(file_paths)} documents:\n{file_list}\n\n"
        if processing_type == "combined":
            return base + "Merge the content into a single coherent markdown document, preserving structure."
        if processing_type == "individual":
            return base + "Convert each document to markdown under its own heading."
        if processing_type == "summary":
            return base + "Create an EXECUTIVE SUMMARY followed by detailed markdown conversions per document."
        if processing_type == "comparison":
            return base + "Provide a comparison table of the documents, individual summaries, and cross-document insights."
        # default fallback
        return base

    def _format_batch_output(self, response_text: str, file_paths: List[Path], processing_type: str, original_filenames: Optional[List[str]] = None) -> str:
        names = original_filenames if original_filenames else [p.name for p in file_paths]
        header = (
             f"<!-- Multi-Document Processing Results -->\n"
             f"<!-- Processing Type: {processing_type} -->\n"
             f"<!-- Files Processed: {len(file_paths)} -->\n"
             f"<!-- File Names: {', '.join(names)} -->\n\n"
         )
        # Ensure response_text is a string to avoid TypeError when it is None
        safe_resp = "" if response_text is None else str(response_text)
        return header + safe_resp

    def _convert_batch_with_docling(self, paths: List[Path], ocr_method: Optional[str], **kwargs) -> List[str]:
        """Run Docling conversion on a list of Paths and return markdown list."""
        if self._check_cancellation():
            raise DocumentProcessingError("Conversion cancelled")

        # Create CPU converter for batch processing (GPU not supported for batch yet)
        converter = self._create_cpu_converter(ocr_method, **kwargs)

        # Convert all docs
        from docling.datamodel.base_models import ConversionStatus
        markdown_results: List[str] = []
        conv_results = converter.convert_all([str(p) for p in paths], raises_on_error=False)

        for idx, conv_res in enumerate(conv_results):
            if self._check_cancellation():
                raise DocumentProcessingError("Conversion cancelled")

            if conv_res.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                markdown_results.append(conv_res.document.export_to_markdown())
            else:
                raise DocumentProcessingError(f"Docling failed to convert {paths[idx].name}")
        return markdown_results

    def parse_multiple(
        self,
        file_paths: List[Union[str, Path]],
        processing_type: str = "combined",
        original_filenames: Optional[List[str]] = None,
        ocr_method: Optional[str] = None,
        output_format: str = "markdown",
        **kwargs,
    ) -> str:
        """Multi-document processing using Docling + optional Gemini summarisation/comparison."""
        if not HAS_DOCLING:
            raise ParserError("Docling package not installed")

        paths = [Path(p) for p in file_paths]
        self._validate_batch_files(paths)

        # Run Docling conversion
        markdown_list = self._convert_batch_with_docling(paths, ocr_method, **kwargs)

        # LOCAL composition for combined/individual
        if processing_type in ("combined", "individual"):
            if processing_type == "individual":
                names = original_filenames if original_filenames else [p.name for p in paths]
                sections = [f"# Document {i+1}: {n}\n\n{md}" for i, (n, md) in enumerate(zip(names, markdown_list), 1)]
                combined = "\n\n---\n\n".join(sections)
            else:
                combined = "\n\n---\n\n".join(markdown_list)
            return self._format_batch_output(combined, paths, processing_type, original_filenames)

        # SUMMARY / COMPARISON → Gemini 2.5 Flash
        if not HAS_GEMINI or not config.api.google_api_key:
            raise DocumentProcessingError("Gemini API not available for summary/comparison post-processing")

        prompt = self._create_batch_prompt(paths, processing_type, original_filenames)
        combined_md = "\n\n---\n\n".join(markdown_list)

        try:
            client = genai.Client(api_key=config.api.google_api_key)
            response = client.models.generate_content(
                model=config.model.gemini_model,
                contents=[prompt + "\n\n" + combined_md],
                config={
                    "temperature": config.model.temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": config.model.max_tokens,
                },
            )
            
            # Debug logging for response structure
            logger.debug(f"Gemini response type: {type(response)}")
            logger.debug(f"Gemini response attributes: {dir(response)}")
            
            # Try different ways to extract text from response
            final_text = None
            if hasattr(response, "text") and response.text:
                final_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                # Try to get text from first candidate
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        final_text = candidate.content.parts[0].text
                    elif hasattr(candidate.content, "text"):
                        final_text = candidate.content.text
                elif hasattr(candidate, "text"):
                    final_text = candidate.text
            elif hasattr(response, "content") and response.content:
                final_text = str(response.content)
            
            if not final_text:
                logger.error(f"No text found in Gemini response. Response: {response}")
                raise DocumentProcessingError("Gemini post-processing returned no text")
                
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise DocumentProcessingError(f"Gemini post-processing failed: {str(e)}")

        return self._format_batch_output(final_text, paths, processing_type, original_filenames)


# Register the parser with the registry if available
if HAS_DOCLING:
    ParserRegistry.register(DoclingParser)
    logger.info("Docling parser registered successfully")
else:
    logger.warning("Could not register Docling parser: Package not installed")