# Import spaces module for ZeroGPU support - Must be first import
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

from pathlib import Path
import os
import logging
import sys
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union
import copy
import pickle

from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry

# Import latex2markdown for conversion - No longer needed, using Gemini API
# import latex2markdown

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Constants
MODEL_NAME = "stepfun-ai/GOT-OCR-2.0-hf"
STOP_STR = "<|im_end|>"

class GotOcrParser(DocumentParser):
    """Parser implementation using GOT-OCR 2.0 for document text extraction using transformers.
    
    This implementation uses the transformers model directly for better integration with
    ZeroGPU and avoids subprocess complexity.
    """
    
    # Class variables to hold model information only (not the actual model)
    _model_loaded = False
    
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
            import torch
            import transformers
            
            # Only check if the modules are importable, DO NOT use torch.cuda here
            # as it would initialize CUDA in the main process
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """Parse a document using GOT-OCR 2.0.
        
        Args:
            file_path: Path to the image file
            ocr_method: OCR method to use ('plain', 'format')
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Extracted text from the image, converted to Markdown if formatted
        """
        # Verify dependencies are installed without initializing CUDA
        if not self._check_dependencies():
            raise ImportError(
                "Required dependencies are missing. Please install: "
                "torch transformers"
            )
        
        # Validate file path and extension
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            raise ValueError(
                f"GOT-OCR only supports JPG and PNG formats. "
                f"Received file with extension: {file_path.suffix}"
            )
        
        # Determine OCR mode based on method
        use_format = ocr_method == "format"
        
        # Log the OCR method being used
        logger.info(f"Using OCR method: {ocr_method or 'plain'}")
        
        # Filter kwargs to remove any objects that can't be pickled (like thread locks)
        safe_kwargs = {}
        for key, value in kwargs.items():
            # Skip thread locks and unpicklable objects
            if not key.startswith('_') and not isinstance(value, type):
                try:
                    # Test if it can be copied - this helps identify unpicklable objects
                    copy.deepcopy(value)
                    safe_kwargs[key] = value
                except (TypeError, pickle.PickleError):
                    logger.warning(f"Skipping unpicklable kwarg: {key}")
        
        # Process the image using transformers
        try:
            # Use the spaces.GPU decorator if available
            if HAS_SPACES:
                # Use string path instead of Path object for better pickling
                image_path_str = str(file_path)
                
                # Call the wrapper function that handles ZeroGPU safely
                return self._safe_gpu_process(image_path_str, use_format, **safe_kwargs)
            else:
                # Fallback for environments without spaces
                return self._process_image_without_gpu(
                    str(file_path), 
                    use_format=use_format,
                    **safe_kwargs
                )
            
        except Exception as e:
            logger.error(f"Error processing image with GOT-OCR: {str(e)}")
            
            # Handle specific errors with helpful messages
            error_type = type(e).__name__
            if error_type == 'OutOfMemoryError':
                raise RuntimeError(
                    "GPU out of memory while processing with GOT-OCR. "
                    "Try using a smaller image or a different parser."
                )
            elif "bfloat16" in str(e):
                raise RuntimeError(
                    "CUDA device does not support bfloat16. This is a known issue with some GPUs. "
                    "Please try using a different parser or contact support."
                )
            elif "CUDA must not be initialized" in str(e):
                raise RuntimeError(
                    "CUDA initialization error. This is likely due to model loading in the main process. "
                    "In ZeroGPU environments, CUDA must only be initialized within @spaces.GPU decorated functions."
                )
            elif "cannot pickle" in str(e):
                raise RuntimeError(
                    f"Serialization error with ZeroGPU: {str(e)}. "
                    "This may be due to thread locks or other unpicklable objects being passed."
                )
            
            # Generic error
            raise RuntimeError(f"Error processing document with GOT-OCR: {str(e)}")
    
    def _safe_gpu_process(self, image_path: str, use_format: bool, **kwargs):
        """Safe wrapper for GPU processing to avoid pickle issues with thread locks."""
        import pickle
        
        try:
            # Call the GPU-decorated function with minimal, picklable arguments
            return self._process_image_with_gpu(image_path, use_format)
        except pickle.PickleError as e:
            logger.error(f"Pickle error in ZeroGPU processing: {str(e)}")
            # Fall back to CPU processing if pickling fails
            logger.warning("Falling back to CPU processing due to pickling error")
            return self._process_image_without_gpu(image_path, use_format=use_format)
    
    def _process_image_without_gpu(self, image_path: str, use_format: bool = False, **kwargs) -> str:
        """Process an image with GOT-OCR model when not using ZeroGPU."""
        logger.warning("ZeroGPU not available. Using direct model loading, which may not work in Spaces.")
        
        # Import here to avoid CUDA initialization in main process
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.image_utils import load_image
        
        # Load the image
        image = load_image(image_path)
        
        # Load processor and model
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        # Use CPU if in main process to avoid CUDA initialization issues
        device = "cpu"
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME, 
            low_cpu_mem_usage=True,
            device_map=device
        )
        model = model.eval()
        
        # Process the image based on the selected OCR method
        if use_format:
            # Format mode
            inputs = processor([image], return_tensors="pt", format=True)
            # Keep on CPU to avoid CUDA initialization
            
            # Generate text
            with torch.no_grad():
                generate_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=processor.tokenizer,
                    stop_strings=STOP_STR,
                    max_new_tokens=4096,
                )
            
            # Decode the generated text
            result = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            
            # Return raw LaTeX output - let post-processing handle conversion
            # This allows for more advanced conversion in the integration module
            logger.info("Returning raw LaTeX output for external processing")
            
        else:
            # Plain text mode
            inputs = processor([image], return_tensors="pt")
            
            # Generate text
            with torch.no_grad():
                generate_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=processor.tokenizer,
                    stop_strings=STOP_STR,
                    max_new_tokens=4096,
                )
            
            # Decode the generated text
            result = processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
        
        # Clean up to free memory
        del model
        del processor
        import gc
        gc.collect()
        
        return result.strip()
    
    # Define the GPU-decorated function for ZeroGPU
    if HAS_SPACES:
        @spaces.GPU()  # Use default ZeroGPU allocation timeframe, matching HF implementation
        def _process_image_with_gpu(self, image_path: str, use_format: bool = False) -> str:
            """Process an image with GOT-OCR model using GPU allocation.
            
            IMPORTANT: All model loading and CUDA operations must happen inside this method.
            NOTE: Function must receive only picklable arguments (no thread locks, etc).
            """
            logger.info("Processing with ZeroGPU allocation")
            
            # Imports inside the GPU-decorated function
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            from transformers.image_utils import load_image
            
            # Load the image
            image = load_image(image_path)
            
            # Now we can load the model inside the GPU-decorated function
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading GOT-OCR model from {MODEL_NAME} on {device}")
            
            # Load processor
            processor = AutoProcessor.from_pretrained(MODEL_NAME)
            
            # Load model
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_NAME, 
                low_cpu_mem_usage=True,
                device_map=device
            )
            
            # Set model to evaluation mode
            model = model.eval()
            
            # Process the image with the model based on the selected OCR method
            if use_format:
                # Format mode (for LaTeX, etc.)
                inputs = processor([image], return_tensors="pt", format=True)
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Generate text
                with torch.no_grad():
                    generate_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=processor.tokenizer,
                        stop_strings=STOP_STR,
                        max_new_tokens=4096,
                    )
                
                # Decode the generated text
                result = processor.decode(
                    generate_ids[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                
                # Return raw LaTeX output - let post-processing handle conversion
                # This allows for more advanced conversion in the integration module
                logger.info("Returning raw LaTeX output for external processing")
            else:
                # Plain text mode
                inputs = processor([image], return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Generate text
                with torch.no_grad():
                    generate_ids = model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=processor.tokenizer,
                        stop_strings=STOP_STR,
                        max_new_tokens=4096,
                    )
                
                # Decode the generated text
                result = processor.decode(
                    generate_ids[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
            
            # Clean up the result
            if result.endswith(STOP_STR):
                result = result[:-len(STOP_STR)]
            
            # Clean up to free memory
            del model
            del processor
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            
            return result.strip()
    else:
        # Define a dummy method if spaces is not available
        def _process_image_with_gpu(self, image_path: str, use_format: bool = False) -> str:
            # This should never be called if HAS_SPACES is False
            return self._process_image_without_gpu(
                image_path, 
                use_format=use_format
            )
    
    @classmethod
    def release_model(cls):
        """Release model resources - not needed with new implementation."""
        logger.info("Model resources managed by ZeroGPU decorator")

# Try to register the parser
try:
    # Only check basic imports, no CUDA initialization
    import torch
    import transformers
    import pickle  # Import pickle for serialization error handling
    ParserRegistry.register(GotOcrParser)
    logger.info("GOT-OCR parser registered successfully")
except ImportError as e:
    logger.warning(f"Could not register GOT-OCR parser: {str(e)}") 