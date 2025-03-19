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

from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry

# Import latex2markdown for conversion
import latex2markdown

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class GotOcrParser(DocumentParser):
    """Parser implementation using GOT-OCR 2.0 for document text extraction using transformers.
    
    This implementation uses the transformers model directly for better integration with
    ZeroGPU and avoids subprocess complexity.
    """
    
    # Class variables to hold model and processor
    _model = None
    _processor = None
    
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
            },
            {
                "id": "box",
                "name": "OCR with Bounding Box",
                "default_params": {"box": "[100,100,200,200]"}  # Default box coordinates
            },
            {
                "id": "color",
                "name": "OCR with Color Filter",
                "default_params": {"color": "red"}  # Default color filter
            },
            {
                "id": "multi_crop",
                "name": "Multi-crop OCR",
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
            
            # Check CUDA availability if using torch
            if hasattr(torch, 'cuda') and not torch.cuda.is_available():
                logger.warning("CUDA is not available. GOT-OCR performs best with GPU acceleration.")
            
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
    
    @classmethod
    def _load_model(cls) -> bool:
        """Load the GOT-OCR model if it's not already loaded."""
        if cls._model is not None and cls._processor is not None:
            return True
        
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            
            # Define the model name - using the HF model ID
            model_name = "stepfun-ai/GOT-OCR-2.0-hf"
            
            # Get the device (CUDA if available, otherwise CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading GOT-OCR model from {model_name}")
            
            # Load processor
            cls._processor = AutoProcessor.from_pretrained(model_name)
            
            # Load model
            cls._model = AutoModelForImageTextToText.from_pretrained(
                model_name, 
                low_cpu_mem_usage=True,
                device_map=device
            )
            
            # Set model to evaluation mode
            cls._model = cls._model.eval()
            
            logger.info("GOT-OCR model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load GOT-OCR model: {str(e)}")
            return False
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """Parse a document using GOT-OCR 2.0.
        
        Args:
            file_path: Path to the image file
            ocr_method: OCR method to use ('plain', 'format', 'box', 'color', 'multi_crop')
            **kwargs: Additional arguments to pass to the model
                - box: For 'box' method, specify box coordinates [x1,y1,x2,y2]
                - color: For 'color' method, specify color ('red', 'green', 'blue')
            
        Returns:
            Extracted text from the image, converted to Markdown if formatted
        """
        # Verify dependencies are installed
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
        
        # Determine OCR mode and parameters based on method
        use_format = ocr_method == "format"
        use_box = ocr_method == "box"
        use_color = ocr_method == "color"
        use_multi_crop = ocr_method == "multi_crop"
        
        # Log the OCR method being used
        logger.info(f"Using OCR method: {ocr_method or 'plain'}")
        
        # Load the model if it's not already loaded
        if not self._load_model():
            raise RuntimeError("Failed to load GOT-OCR model")
        
        # Process the image using transformers
        try:
            # Use the spaces.GPU decorator if available
            if HAS_SPACES:
                return self._process_image_with_gpu(
                    str(file_path), 
                    use_format=use_format,
                    use_box=use_box,
                    use_color=use_color,
                    use_multi_crop=use_multi_crop,
                    **kwargs
                )
            else:
                return self._process_image(
                    str(file_path), 
                    use_format=use_format,
                    use_box=use_box,
                    use_color=use_color,
                    use_multi_crop=use_multi_crop,
                    **kwargs
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
            
            # Generic error
            raise RuntimeError(f"Error processing document with GOT-OCR: {str(e)}")
    
    def _process_image(self, image_path: str, use_format: bool = False, use_box: bool = False, use_color: bool = False, use_multi_crop: bool = False, **kwargs) -> str:
        """Process an image with GOT-OCR model (no GPU decorator)."""
        try:
            from transformers.image_utils import load_image
            import torch
            
            # Load the image
            image = load_image(image_path)
            
            # Define stop string
            stop_str = "<|im_end|>"
            
            # Process the image with the model based on the selected OCR method
            if use_format:
                # Format mode (for LaTeX, etc.)
                inputs = self._processor([image], return_tensors="pt", format=True)
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Generate text
                with torch.no_grad():
                    generate_ids = self._model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=self._processor.tokenizer,
                        stop_strings=stop_str,
                        max_new_tokens=4096,
                    )
                
                # Decode the generated text
                result = self._processor.decode(
                    generate_ids[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                
                # Convert to Markdown if it's formatted
                l2m = latex2markdown.LaTeX2Markdown(result)
                result = l2m.to_markdown()
                
            elif use_box:
                # Box-based OCR
                box_coords = kwargs.get('box', '[100,100,200,200]')
                if isinstance(box_coords, str):
                    # Convert string representation to list if needed
                    import json
                    try:
                        box_coords = json.loads(box_coords.replace("'", '"'))
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid box format: {box_coords}. Using default.")
                        box_coords = [100, 100, 200, 200]
                
                logger.info(f"Using box coordinates: {box_coords}")
                
                # Process with box parameter
                inputs = self._processor([image], return_tensors="pt", box=box_coords)
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Generate text
                with torch.no_grad():
                    generate_ids = self._model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=self._processor.tokenizer,
                        stop_strings=stop_str,
                        max_new_tokens=4096,
                    )
                
                # Decode the generated text
                result = self._processor.decode(
                    generate_ids[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
            
            elif use_color:
                # Color-based OCR
                color = kwargs.get('color', 'red')
                logger.info(f"Using color filter: {color}")
                
                # Process with color parameter
                inputs = self._processor([image], return_tensors="pt", color=color)
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Generate text
                with torch.no_grad():
                    generate_ids = self._model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=self._processor.tokenizer,
                        stop_strings=stop_str,
                        max_new_tokens=4096,
                    )
                
                # Decode the generated text
                result = self._processor.decode(
                    generate_ids[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
            
            elif use_multi_crop:
                # Multi-crop OCR
                logger.info("Using multi-crop OCR")
                
                # Process with multi-crop parameter
                inputs = self._processor(
                    [image],
                    return_tensors="pt",
                    format=True,
                    crop_to_patches=True,
                    max_patches=5,
                )
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Generate text
                with torch.no_grad():
                    generate_ids = self._model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=self._processor.tokenizer,
                        stop_strings=stop_str,
                        max_new_tokens=4096,
                    )
                
                # Decode the generated text
                result = self._processor.decode(
                    generate_ids[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                
                # Convert to Markdown as multi-crop uses format
                l2m = latex2markdown.LaTeX2Markdown(result)
                result = l2m.to_markdown()
                
            else:
                # Plain text mode
                inputs = self._processor([image], return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                # Generate text
                with torch.no_grad():
                    generate_ids = self._model.generate(
                        **inputs,
                        do_sample=False,
                        tokenizer=self._processor.tokenizer,
                        stop_strings=stop_str,
                        max_new_tokens=4096,
                    )
                
                # Decode the generated text
                result = self._processor.decode(
                    generate_ids[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
            
            # Clean up the result
            if result.endswith(stop_str):
                result = result[:-len(stop_str)]
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error in _process_image: {str(e)}")
            raise
    
    # Define the GPU-decorated function for ZeroGPU
    if HAS_SPACES:
        @spaces.GPU(duration=180)  # Allocate up to 3 minutes for OCR processing
        def _process_image_with_gpu(self, image_path: str, use_format: bool = False, use_box: bool = False, use_color: bool = False, use_multi_crop: bool = False, **kwargs) -> str:
            """Process an image with GOT-OCR model using GPU allocation."""
            logger.info("Processing with ZeroGPU allocation")
            return self._process_image(
                image_path, 
                use_format=use_format,
                use_box=use_box,
                use_color=use_color,
                use_multi_crop=use_multi_crop,
                **kwargs
            )
    else:
        # Define a dummy method if spaces is not available
        def _process_image_with_gpu(self, image_path: str, use_format: bool = False, use_box: bool = False, use_color: bool = False, use_multi_crop: bool = False, **kwargs) -> str:
            # This should never be called if HAS_SPACES is False
            return self._process_image(
                image_path, 
                use_format=use_format,
                use_box=use_box,
                use_color=use_color,
                use_multi_crop=use_multi_crop,
                **kwargs
            )
    
    @classmethod
    def release_model(cls):
        """Release the model resources."""
        if cls._model is not None:
            # Clear the model from memory
            cls._model = None
            cls._processor = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared")
            except ImportError:
                pass
            
            logger.info("GOT-OCR model resources released")

# Try to register the parser
try:
    # Only check basic imports, detailed dependency check happens in parse method
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    ParserRegistry.register(GotOcrParser)
    logger.info("GOT-OCR parser registered successfully")
except ImportError as e:
    logger.warning(f"Could not register GOT-OCR parser: {str(e)}") 