from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import os
import sys

# Set PyTorch environment variables for T4 compatibility
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0+PTX"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_AMP_AUTOCAST_DTYPE"] = "float16"

from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry

# Configure logging
logger = logging.getLogger(__name__)

class GotOcrParser(DocumentParser):
    """Parser implementation using GOT-OCR 2.0 for document text extraction.
    Optimized for NVIDIA T4 GPUs with explicit float16 support.
    """
    
    _model = None
    _processor = None
    _stop_str = "<|im_end|>"
    
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
    def _check_dependencies(cls) -> bool:
        """Check if all required dependencies are installed."""
        try:
            import numpy
            import torch
            import transformers
            import tiktoken
            
            # Check CUDA availability if using torch
            if hasattr(torch, 'cuda') and not torch.cuda.is_available():
                logger.warning("CUDA is not available. GOT-OCR performs best with GPU acceleration.")
            
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
    
    @classmethod
    def _load_model(cls):
        """Load the GOT-OCR model and tokenizer if not already loaded."""
        if cls._model is None or cls._processor is None:
            try:
                # Import dependencies inside the method to avoid global import errors
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor
                
                logger.info("Loading GOT-OCR model and processor...")
                
                # Determine device
                device_map = 'cuda' if torch.cuda.is_available() else 'auto'
                if device_map == 'cuda':
                    logger.info("Using CUDA for model inference")
                else:
                    logger.warning("Using CPU for model inference (not recommended)")
                
                # Load the processor (includes tokenizer)
                cls._processor = AutoProcessor.from_pretrained(
                    'stepfun-ai/GOT-OCR2_0-hf'
                )
                
                # Load model with explicit float16 for T4 compatibility
                cls._model = AutoModelForImageTextToText.from_pretrained(
                    'stepfun-ai/GOT-OCR2_0-hf',
                    low_cpu_mem_usage=True,
                    device_map=device_map,
                    torch_dtype=torch.float16  # Force float16 for T4 compatibility
                )
                
                # Explicitly convert model to half precision (float16)
                cls._model = cls._model.half().eval()
                
                # Move to CUDA if available
                if device_map == 'cuda':
                    cls._model = cls._model.cuda()
                
                # Patch torch.autocast to force float16 instead of bfloat16
                original_autocast = torch.autocast
                def patched_autocast(*args, **kwargs):
                    # Force dtype to float16 when CUDA is involved
                    if args and args[0] == "cuda":
                        kwargs['dtype'] = torch.float16
                    return original_autocast(*args, **kwargs)
                
                # Apply the patch
                torch.autocast = patched_autocast
                logger.info("Patched torch.autocast to always use float16 for CUDA operations")
                
                logger.info("GOT-OCR model loaded successfully")
                return True
            except Exception as e:
                cls._model = None
                cls._processor = None
                logger.error(f"Failed to load GOT-OCR model: {str(e)}")
                return False
        return True
    
    @classmethod
    def release_model(cls):
        """Release the model from memory."""
        try:
            import torch
            
            if cls._model is not None:
                del cls._model
                cls._model = None
            
            if cls._processor is not None:
                del cls._processor
                cls._processor = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("GOT-OCR model released from memory")
        except Exception as e:
            logger.error(f"Error releasing model: {str(e)}")
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """Parse a document using GOT-OCR 2.0.
        
        Args:
            file_path: Path to the image file
            ocr_method: OCR method to use ('plain' or 'format')
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Extracted text from the image
        """
        # Verify dependencies are installed
        if not self._check_dependencies():
            raise ImportError(
                "Required dependencies are missing. Please install: "
                "torch==2.0.1 torchvision==0.15.2 transformers==4.37.2 "
                "tiktoken==0.6.0 verovio==4.3.1 accelerate==0.28.0"
            )
        
        # Load model if not already loaded
        if not self._load_model():
            raise RuntimeError("Failed to load GOT-OCR model")
        
        # Import torch here to ensure it's available
        import torch
        from transformers.image_utils import load_image
        
        # Validate file path and extension
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            raise ValueError(
                f"GOT-OCR only supports JPG and PNG formats. "
                f"Received file with extension: {file_path.suffix}"
            )
        
        # Determine format flag based on OCR method
        format_flag = ocr_method == "format"
        logger.info(f"Using OCR method: {'format' if format_flag else 'plain'}")
        
        # Process the image
        try:
            logger.info(f"Processing image with GOT-OCR: {file_path}")
            
            # Load image with transformers utils
            image = load_image(str(file_path))
            
            # First attempt: Normal processing with autocast
            try:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # Process image with format flag if needed
                    if format_flag:
                        inputs = self._processor(image, return_tensors="pt", format=True).to("cuda")
                    else:
                        inputs = self._processor(image, return_tensors="pt").to("cuda")
                    
                    # Generate text
                    generate_ids = self._model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=self._processor.tokenizer,
                        stop_strings=self._stop_str,
                        max_new_tokens=4096,
                    )
                    
                    # Decode the generated text
                    result = self._processor.decode(
                        generate_ids[0, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    
                    return result
                    
            except RuntimeError as e:
                # Check if it's a bfloat16 error
                if "bfloat16" in str(e) or "BFloat16" in str(e):
                    logger.warning("Encountered bfloat16 error, trying float16 fallback")
                    
                    # Second attempt: More aggressive float16 forcing
                    try:
                        # Ensure model is float16
                        self._model = self._model.half()
                        
                        # Set default dtype temporarily
                        original_dtype = torch.get_default_dtype()
                        torch.set_default_dtype(torch.float16)
                        
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            # Process image with format flag if needed
                            if format_flag:
                                inputs = self._processor(image, return_tensors="pt", format=True).to("cuda")
                            else:
                                inputs = self._processor(image, return_tensors="pt").to("cuda")
                            
                            # Generate text
                            generate_ids = self._model.generate(
                                **inputs,
                                do_sample=False,
                                tokenizer=self._processor.tokenizer,
                                stop_strings=self._stop_str,
                                max_new_tokens=4096,
                            )
                            
                            # Decode the generated text
                            result = self._processor.decode(
                                generate_ids[0, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True,
                            )
                        
                        # Restore default dtype
                        torch.set_default_dtype(original_dtype)
                        return result
                    except Exception as inner_e:
                        logger.error(f"Float16 fallback failed: {str(inner_e)}")
                        raise RuntimeError(
                            f"Failed to process image with GOT-OCR: {str(inner_e)}"
                        )
                else:
                    # Not a bfloat16 error, re-raise
                    raise
                    
        except Exception as e:
            logger.error(f"Error processing image with GOT-OCR: {str(e)}")
            
            # Handle specific errors with helpful messages
            error_type = type(e).__name__
            if error_type == 'OutOfMemoryError':
                self.release_model()
                raise RuntimeError(
                    "GPU out of memory while processing with GOT-OCR. "
                    "Try using a smaller image or a different parser."
                )
            
            # Generic error
            raise RuntimeError(f"Error processing document with GOT-OCR: {str(e)}")

# Try to register the parser
try:
    # Only check basic imports, detailed dependency check happens in parse method
    import numpy
    import torch
    ParserRegistry.register(GotOcrParser)
    logger.info("GOT-OCR parser registered successfully")
except ImportError as e:
    logger.warning(f"Could not register GOT-OCR parser: {str(e)}") 