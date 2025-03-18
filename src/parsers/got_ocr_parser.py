from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import os
import tempfile
import logging
import sys
import importlib

from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry

# Configure logging
logger = logging.getLogger(__name__)

# Global flag for NumPy availability
NUMPY_AVAILABLE = False
NUMPY_VERSION = None

# Initialize torch as None in global scope to prevent reference errors
torch = None
GOT_AVAILABLE = False

# Try to import NumPy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    NUMPY_VERSION = np.__version__
    logger.info(f"NumPy version {NUMPY_VERSION} is available")
except ImportError:
    NUMPY_AVAILABLE = False
    logger.error("NumPy is not available. This is required for GOT-OCR.")

# Check if required packages are installed
try:    
    import torch as torch_module
    torch = torch_module  # Assign to global variable
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
except ImportError as e:
    GOT_AVAILABLE = False
    logger.warning(f"GOT-OCR dependencies not installed: {str(e)}. The parser will not be available.")

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
        global NUMPY_AVAILABLE, torch
        
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is not available. This is required for GOT-OCR.")
            
        if torch is None:
            raise ImportError("PyTorch is not available. This is required for GOT-OCR.")
            
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
                
                # Set torch default dtype to float16 since the CUDA device doesn't support bfloat16
                logger.info("Setting default tensor type to float16")
                torch.set_default_tensor_type(torch.FloatTensor)
                torch.set_default_dtype(torch.float16)
                
                cls._model = AutoModel.from_pretrained(
                    'stepfun-ai/GOT-OCR2_0', 
                    trust_remote_code=True, 
                    low_cpu_mem_usage=True, 
                    device_map=device_map, 
                    use_safetensors=True,
                    pad_token_id=cls._tokenizer.eos_token_id,
                    torch_dtype=torch.float16  # Explicitly specify float16 dtype
                )
                
                # Set model to evaluation mode
                if device_map == 'cuda':
                    cls._model = cls._model.eval().cuda()
                else:
                    cls._model = cls._model.eval()
                
                # Reset default dtype to float32 after model loading
                torch.set_default_dtype(torch.float32)
                torch.set_default_tensor_type(torch.FloatTensor)
                    
                logger.info("GOT-OCR model loaded successfully")
            except Exception as e:
                cls._model = None
                cls._tokenizer = None
                logger.error(f"Failed to load GOT-OCR model: {str(e)}")
                raise RuntimeError(f"Failed to load GOT-OCR model: {str(e)}")
    
    @classmethod
    def release_model(cls):
        """Release the model from memory."""
        global torch
        
        if cls._model is not None:
            del cls._model
            cls._model = None
        if cls._tokenizer is not None:
            del cls._tokenizer
            cls._tokenizer = None
        if torch is not None and hasattr(torch, 'cuda') and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        logger.info("GOT-OCR model released from memory")
    
    def _try_install_numpy(self):
        """Attempt to install NumPy using pip."""
        global NUMPY_AVAILABLE, NUMPY_VERSION
        
        logger.warning("Attempting to install NumPy...")
        try:
            import subprocess
            # Try to install numpy with explicit version constraint for compatibility with torchvision
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "numpy<2.0.0", "--no-cache-dir"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"NumPy installation result: {result.stdout}")
            
            # Try to import numpy again
            importlib.invalidate_caches()
            import numpy as np
            importlib.reload(np)
            
            NUMPY_AVAILABLE = True
            NUMPY_VERSION = np.__version__
            logger.info(f"NumPy installed successfully: version {NUMPY_VERSION}")
            return True
        except Exception as e:
            logger.error(f"Failed to install NumPy: {str(e)}")
            if hasattr(e, 'stderr'):
                logger.error(f"Installation error output: {e.stderr}")
            return False
    
    def _try_install_torch(self):
        """Attempt to install PyTorch using pip."""
        global torch
        
        logger.warning("Attempting to install PyTorch...")
        try:
            import subprocess
            # Install PyTorch with version constraint as per the requirements
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "torch==2.0.1", "torchvision==0.15.2", "--no-cache-dir"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"PyTorch installation result: {result.stdout}")
            
            # Try to import torch again
            importlib.invalidate_caches()
            import torch as torch_module
            torch = torch_module
            
            logger.info(f"PyTorch installed successfully: version {torch.__version__}")
            return True
        except Exception as e:
            logger.error(f"Failed to install PyTorch: {str(e)}")
            if hasattr(e, 'stderr'):
                logger.error(f"Installation error output: {e.stderr}")
            return False
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """Parse a document using GOT-OCR 2.0."""
        global NUMPY_AVAILABLE, GOT_AVAILABLE, torch
        
        # Check NumPy availability and try to install if not available
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available, attempting to install it...")
            if self._try_install_numpy():
                # NumPy is now available
                logger.info("NumPy is now available")
            else:
                logger.error("Failed to install NumPy. Cannot proceed with GOT-OCR.")
                raise ImportError(
                    "NumPy is not available and could not be installed automatically. "
                    "Please ensure NumPy is installed in your environment. "
                    "Add the following to your logs for debugging: NUMPY_INSTALLATION_FAILED"
                )
        
        # Check PyTorch availability and try to install if not available
        if torch is None:
            logger.warning("PyTorch not available, attempting to install it...")
            if self._try_install_torch():
                # PyTorch is now available
                logger.info("PyTorch is now available")
            else:
                logger.error("Failed to install PyTorch. Cannot proceed with GOT-OCR.")
                raise ImportError(
                    "PyTorch is not available and could not be installed automatically. "
                    "Please ensure PyTorch is installed in your environment."
                )
        
        # Update GOT availability flag after potential installations
        try:
            if NUMPY_AVAILABLE and torch is not None:
                import transformers
                GOT_AVAILABLE = True
                logger.info("Updated GOT availability after installations: Available")
            else:
                GOT_AVAILABLE = False
                logger.error("GOT availability after installations: Not Available (missing dependencies)")
        except ImportError:
            GOT_AVAILABLE = False
            logger.error("Transformers not available. GOT-OCR cannot be used.")
        
        # Check overall GOT availability
        if not GOT_AVAILABLE:
            if not NUMPY_AVAILABLE:
                logger.error("NumPy is still not available after installation attempt.")
                raise ImportError(
                    "NumPy is not available. This is required for GOT-OCR. "
                    "Please ensure NumPy is installed in your environment. "
                    "Environment details: Python " + sys.version
                )
            elif torch is None:
                logger.error("PyTorch is still not available after installation attempt.")
                raise ImportError(
                    "PyTorch is not available. This is required for GOT-OCR. "
                    "Please ensure PyTorch is installed in your environment."
                )
            else:
                logger.error("Other GOT-OCR dependencies missing even though NumPy and PyTorch are available.")
                raise ImportError(
                    "GOT-OCR dependencies not installed. Please install required packages: "
                    "transformers, tiktoken, verovio, accelerate"
                )
        
        # Check if CUDA is available
        cuda_available = torch is not None and hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available') and torch.cuda.is_available()
        if not cuda_available:
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
            # Check if numpy needs to be reloaded
            if 'numpy' in sys.modules:
                logger.info("NumPy module found in sys.modules, attempting to reload...")
                try:
                    importlib.reload(sys.modules['numpy'])
                    import numpy as np
                    logger.info(f"NumPy reloaded successfully: version {np.__version__}")
                except Exception as e:
                    logger.error(f"Error reloading NumPy: {str(e)}")
            
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
                
        except Exception as e:
            error_type = type(e).__name__
            
            # Handle specific error types
            if torch is not None and hasattr(torch, 'cuda') and error_type == 'OutOfMemoryError':
                self.release_model()  # Release memory
                logger.error("GPU out of memory while processing with GOT-OCR")
                raise RuntimeError(
                    "GPU out of memory while processing with GOT-OCR. "
                    "Try using a smaller image or a different parser."
                )
            elif error_type == 'AttributeError' and "get_max_length" in str(e):
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

# Register the parser with the registry if dependencies are available
try:
    if NUMPY_AVAILABLE and torch is not None:
        ParserRegistry.register(GotOcrParser)
        logger.info("GOT-OCR parser registered successfully")
    else:
        missing_deps = []
        if not NUMPY_AVAILABLE:
            missing_deps.append("NumPy")
        if torch is None:
            missing_deps.append("PyTorch")
        logger.warning(f"GOT-OCR parser not registered: missing dependencies: {', '.join(missing_deps)}")
except Exception as e:
    logger.error(f"Error registering GOT-OCR parser: {str(e)}") 