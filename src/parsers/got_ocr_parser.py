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
    
    GOT_AVAILABLE = True and NUMPY_AVAILABLE
except ImportError:
    GOT_AVAILABLE = False
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
    def get_description(cls) -> str:
        return "GOT-OCR 2.0 parser for converting images to text (requires CUDA)"
    
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
            
            # Return the result directly as markdown
            return result
                
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

# Register the parser with the registry if GOT is available
if GOT_AVAILABLE:
    ParserRegistry.register(GotOcrParser)
    logger.info("GOT-OCR parser registered successfully")
else:
    logger.warning("GOT-OCR parser not registered: required dependencies not installed") 