from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import os
import tempfile
import logging

from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry

# Configure logging
logger = logging.getLogger(__name__)

# Check if required packages are installed
try:
    # First check for numpy as it's required by torch
    try:
        import numpy as np
        NUMPY_AVAILABLE = True
        logger.info(f"NumPy version {np.__version__} is available")
    except ImportError:
        NUMPY_AVAILABLE = False
        logger.error("NumPy is not available. This is required for GOT-OCR.")
    
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    
    # Check if transformers version is compatible
    from packaging import version
    if version.parse(transformers.__version__) >= version.parse("4.48.0"):
        logger.warning(
            f"Transformers version {transformers.__version__} may not be compatible with GOT-OCR. "
            "Consider downgrading to version <4.48.0"
        )
    
    # Import spaces for ZeroGPU support
    try:
        import spaces
        ZEROGPU_AVAILABLE = True
        logger.info("ZeroGPU support is available")
    except ImportError:
        ZEROGPU_AVAILABLE = False
        logger.info("ZeroGPU not available, will use standard GPU if available")
    
    GOT_AVAILABLE = True and NUMPY_AVAILABLE
except ImportError:
    GOT_AVAILABLE = False
    ZEROGPU_AVAILABLE = False
    NUMPY_AVAILABLE = False
    logger.warning("GOT-OCR dependencies not installed. The parser will not be available.")

class GotOcrParser(DocumentParser):
    """Parser implementation using GOT-OCR 2.0."""
    
    _model = None
    _tokenizer = None
    
    @classmethod
    def get_name(cls) -> str:
        return "GOT-OCR (jpg,png only)"
    
    @classmethod
    def get_supported_ocr_methods(cls) -> List[Dict[str, Any]]:
        return [
            {
                "id": "plain",
                "name": "Plain Text",
                "default_params": {}
            },
            {
                "id": "format",
                "name": "Formatted Text",
                "default_params": {}
            }
        ]
    
    @classmethod
    def _load_model(cls):
        """Load the GOT-OCR model and tokenizer if not already loaded."""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is not available. This is required for GOT-OCR.")
            
        if cls._model is None or cls._tokenizer is None:
            try:
                logger.info("Loading GOT-OCR model and tokenizer...")
                cls._tokenizer = AutoTokenizer.from_pretrained(
                    'stepfun-ai/GOT-OCR2_0', 
                    trust_remote_code=True
                )
                
                # Determine device mapping based on CUDA availability
                if torch.cuda.is_available():
                    logger.info("Using CUDA device for model loading")
                    device_map = 'cuda'
                else:
                    logger.warning("No GPU available, falling back to CPU (not recommended)")
                    device_map = 'auto'
                
                cls._model = AutoModel.from_pretrained(
                    'stepfun-ai/GOT-OCR2_0', 
                    trust_remote_code=True, 
                    low_cpu_mem_usage=True, 
                    device_map=device_map, 
                    use_safetensors=True,
                    pad_token_id=cls._tokenizer.eos_token_id
                )
                
                # Set model to evaluation mode
                if device_map == 'cuda':
                    cls._model = cls._model.eval().cuda()
                else:
                    cls._model = cls._model.eval()
                    
                logger.info("GOT-OCR model loaded successfully")
            except Exception as e:
                cls._model = None
                cls._tokenizer = None
                logger.error(f"Failed to load GOT-OCR model: {str(e)}")
                raise RuntimeError(f"Failed to load GOT-OCR model: {str(e)}")
    
    @classmethod
    def release_model(cls):
        """Release the model from memory."""
        if cls._model is not None:
            del cls._model
            cls._model = None
        if cls._tokenizer is not None:
            del cls._tokenizer
            cls._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("GOT-OCR model released from memory")
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """Parse a document using GOT-OCR 2.0."""
        if not GOT_AVAILABLE:
            if not NUMPY_AVAILABLE:
                raise ImportError(
                    "NumPy is not available. This is required for GOT-OCR. "
                    "Please ensure NumPy is installed in your environment."
                )
            else:
                raise ImportError(
                    "GOT-OCR dependencies not installed. Please install required packages: "
                    "torch, transformers, tiktoken, verovio, accelerate"
                )
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.warning("No GPU available. GOT-OCR performance may be severely degraded.")
        
        # Check file extension
        file_path = Path(file_path)
        if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            raise ValueError(
                "GOT-OCR only supports JPG and PNG formats. "
                f"Received file with extension: {file_path.suffix}"
            )
        
        # Determine OCR type based on method
        ocr_type = "format" if ocr_method == "format" else "ocr"
        
        # Use ZeroGPU if available, otherwise use regular processing
        if ZEROGPU_AVAILABLE:
            try:
                return self._parse_with_zerogpu(file_path, ocr_type, **kwargs)
            except RuntimeError as e:
                if "numpy" in str(e).lower():
                    logger.warning("NumPy issues in ZeroGPU environment, falling back to regular processing")
                    return self._parse_regular(file_path, ocr_type, **kwargs)
                else:
                    raise
        else:
            return self._parse_regular(file_path, ocr_type, **kwargs)
    
    def _parse_regular(self, file_path: Path, ocr_type: str, **kwargs) -> str:
        """Regular parsing without ZeroGPU."""
        try:
            # Load the model
            self._load_model()
            
            # Use the model's chat method as shown in the documentation
            logger.info(f"Processing image with GOT-OCR: {file_path}")
            result = self._model.chat(
                self._tokenizer, 
                str(file_path), 
                ocr_type=ocr_type
            )
            
            return self._format_result(result, **kwargs)
                
        except torch.cuda.OutOfMemoryError:
            self.release_model()  # Release memory
            logger.error("GPU out of memory while processing with GOT-OCR")
            raise RuntimeError(
                "GPU out of memory while processing with GOT-OCR. "
                "Try using a smaller image or a different parser."
            )
        except AttributeError as e:
            if "get_max_length" in str(e):
                logger.error(f"Transformers version compatibility error: {str(e)}")
                self.release_model()  # Release memory
                raise RuntimeError(
                    "Transformers version compatibility error with GOT-OCR. "
                    "Please downgrade transformers to version <4.48.0. "
                    f"Error: {str(e)}"
                )
            else:
                logger.error(f"Error processing document with GOT-OCR: {str(e)}")
                raise RuntimeError(f"Error processing document with GOT-OCR: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing document with GOT-OCR: {str(e)}")
            raise RuntimeError(f"Error processing document with GOT-OCR: {str(e)}")
    
    def _parse_with_zerogpu(self, file_path: Path, ocr_type: str, **kwargs) -> str:
        """Parse using ZeroGPU for dynamic GPU allocation."""
        try:
            # Define the GPU-dependent function
            @spaces.GPU
            def process_with_gpu():
                # Ensure NumPy is available
                try:
                    import numpy
                except ImportError:
                    # Try to install numpy if not available
                    import subprocess
                    import sys
                    logger.warning("NumPy not found in ZeroGPU environment, attempting to install...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=1.24.0"])
                    import numpy
                    logger.info(f"NumPy {numpy.__version__} installed successfully in ZeroGPU environment")
                
                # Load the model
                self._load_model()
                
                # Use the model's chat method
                logger.info(f"Processing image with GOT-OCR using ZeroGPU: {file_path}")
                return self._model.chat(
                    self._tokenizer, 
                    str(file_path), 
                    ocr_type=ocr_type
                )
            
            # Call the GPU-decorated function
            result = process_with_gpu()
            
            # Format and return the result
            return self._format_result(result, **kwargs)
            
        except ImportError as e:
            if "numpy" in str(e).lower():
                logger.error(f"NumPy import error in ZeroGPU environment: {str(e)}")
                raise RuntimeError(
                    "NumPy is not available in the ZeroGPU environment. "
                    "This is a known issue with some HuggingFace Spaces. "
                    "Please try using a different parser or contact support."
                )
            else:
                logger.error(f"Import error in ZeroGPU environment: {str(e)}")
                raise RuntimeError(f"Error processing document with GOT-OCR (ZeroGPU): {str(e)}")
        except Exception as e:
            logger.error(f"Error processing document with GOT-OCR (ZeroGPU): {str(e)}")
            raise RuntimeError(f"Error processing document with GOT-OCR (ZeroGPU): {str(e)}")
    
    def _format_result(self, result: str, **kwargs) -> str:
        """Format the OCR result based on the requested format."""
        output_format = kwargs.get("output_format", "markdown").lower()
        if output_format == "json":
            return json.dumps({"content": result}, ensure_ascii=False, indent=2)
        elif output_format == "text":
            # Simple markdown to text conversion
            return result.replace("#", "").replace("*", "").replace("_", "")
        elif output_format == "document_tags":
            return f"<doc>\n{result}\n</doc>"
        else:
            return result

# Register the parser with the registry if GOT is available
if GOT_AVAILABLE:
    ParserRegistry.register(GotOcrParser)
    logger.info("GOT-OCR parser registered successfully")
else:
    logger.warning("GOT-OCR parser not registered: required dependencies not installed") 