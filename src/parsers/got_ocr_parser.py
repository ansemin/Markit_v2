from pathlib import Path
import os
import logging
import sys
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union

from src.parsers.parser_interface import DocumentParser
from src.parsers.parser_registry import ParserRegistry

# Import latex2markdown instead of custom converter
import latex2markdown

# Configure logging
logger = logging.getLogger(__name__)

class GotOcrParser(DocumentParser):
    """Parser implementation using GOT-OCR 2.0 for document text extraction using GitHub repository.
    
    This implementation uses the official GOT-OCR2.0 GitHub repository through subprocess calls
    rather than loading the model directly through Hugging Face Transformers.
    """
    
    # Path to the GOT-OCR repository
    _repo_path = None
    _weights_path = None
    
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
            import tiktoken
            
            # Check CUDA availability if using torch
            if hasattr(torch, 'cuda') and not torch.cuda.is_available():
                logger.warning("CUDA is not available. GOT-OCR performs best with GPU acceleration.")
            
            # Check for latex2markdown
            try:
                import latex2markdown
                logger.info("latex2markdown package found")
            except ImportError:
                logger.warning("latex2markdown package not found. Installing...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "latex2markdown"],
                    check=True
                )
            
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
    
    @classmethod
    def _setup_repository(cls) -> bool:
        """Set up the GOT-OCR2.0 repository if it's not already set up."""
        if cls._repo_path is not None and os.path.exists(cls._repo_path):
            return True
        
        try:
            # Create a temporary directory for the repository
            repo_dir = os.path.join(tempfile.gettempdir(), "GOT-OCR2.0")
            
            # Check if the repository already exists
            if not os.path.exists(repo_dir):
                logger.info("Cloning GOT-OCR2.0 repository...")
                subprocess.run(
                    ["git", "clone", "https://github.com/Ucas-HaoranWei/GOT-OCR2.0.git", repo_dir],
                    check=True
                )
            else:
                logger.info("GOT-OCR2.0 repository already exists, skipping clone")
            
            cls._repo_path = repo_dir
            
            # Set up the weights directory
            weights_dir = os.path.join(repo_dir, "GOT_weights")
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir, exist_ok=True)
            
            cls._weights_path = weights_dir
            
            # Check if weights exist, if not download them
            weight_files = [f for f in os.listdir(weights_dir) if f.endswith(".bin") or f.endswith(".safetensors")]
            if not weight_files:
                logger.info("Downloading GOT-OCR2.0 weights...")
                logger.info("This may take some time depending on your internet connection.")
                logger.info("Downloading from Hugging Face repository...")
                
                # Use Hugging Face CLI to download the model
                subprocess.run(
                    ["huggingface-cli", "download", "stepfun-ai/GOT-OCR2_0", "--local-dir", weights_dir],
                    check=True
                )
                
                # Additional check to verify downloads
                weight_files = [f for f in os.listdir(weights_dir) if f.endswith(".bin") or f.endswith(".safetensors")]
                if not weight_files:
                    logger.error("Failed to download weights. Please download them manually and place in GOT_weights directory.")
                    return False
            
            logger.info("GOT-OCR2.0 repository and weights set up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set up GOT-OCR2.0 repository: {str(e)}")
            return False
    
    def parse(self, file_path: Union[str, Path], ocr_method: Optional[str] = None, **kwargs) -> str:
        """Parse a document using GOT-OCR 2.0.
        
        Args:
            file_path: Path to the image file
            ocr_method: OCR method to use ('plain' or 'format')
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Extracted text from the image, converted to Markdown if formatted
        """
        # Verify dependencies are installed
        if not self._check_dependencies():
            raise ImportError(
                "Required dependencies are missing. Please install: "
                "torch==2.0.1 torchvision==0.15.2 transformers==4.37.2 "
                "tiktoken==0.6.0 verovio==4.3.1 accelerate==0.28.0"
            )
        
        # Set up the repository
        if not self._setup_repository():
            raise RuntimeError("Failed to set up GOT-OCR2.0 repository")
        
        # Validate file path and extension
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            raise ValueError(
                f"GOT-OCR only supports JPG and PNG formats. "
                f"Received file with extension: {file_path.suffix}"
            )
        
        # Determine OCR type based on method
        ocr_type = "format" if ocr_method == "format" else "ocr"
        logger.info(f"Using OCR method: {ocr_type}")
        
        # Check if render is specified in kwargs
        render = kwargs.get('render', False)
        
        # Process the image using the GOT-OCR repository
        try:
            logger.info(f"Processing image with GOT-OCR: {file_path}")
            
            # Create the command for running the GOT-OCR script
            cmd = [
                sys.executable,
                os.path.join(self._repo_path, "GOT", "demo", "run_ocr_2.0.py"),
                "--model-name", self._weights_path,
                "--image-file", str(file_path),
                "--type", ocr_type
            ]
            
            # Add render flag if required
            if render:
                cmd.append("--render")
            
            # Check if box or color is specified in kwargs
            if 'box' in kwargs and kwargs['box']:
                cmd.extend(["--box", str(kwargs['box'])])
            
            if 'color' in kwargs and kwargs['color']:
                cmd.extend(["--color", kwargs['color']])
            
            # Run the command and capture output
            logger.info(f"Running command: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Process the output
            result = process.stdout.strip()
            
            # If render was requested, find and return the path to the HTML file
            if render:
                # The rendered results are in /results/demo.html according to the README
                html_result_path = os.path.join(self._repo_path, "results", "demo.html")
                if os.path.exists(html_result_path):
                    with open(html_result_path, 'r') as f:
                        html_content = f.read()
                    return html_content
                else:
                    logger.warning(f"Rendered HTML file not found at {html_result_path}")
            
            # Check if we need to convert from LaTeX to Markdown
            if ocr_type == "format":
                logger.info("Converting formatted LaTeX output to Markdown using latex2markdown")
                # Use the latex2markdown package instead of custom converter
                l2m = latex2markdown.LaTeX2Markdown(result)
                result = l2m.to_markdown()
            
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running GOT-OCR command: {str(e)}")
            logger.error(f"Stderr: {e.stderr}")
            raise RuntimeError(f"Error processing document with GOT-OCR: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error processing image with GOT-OCR: {str(e)}")
            
            # Handle specific errors with helpful messages
            error_type = type(e).__name__
            if error_type == 'OutOfMemoryError':
                raise RuntimeError(
                    "GPU out of memory while processing with GOT-OCR. "
                    "Try using a smaller image or a different parser."
                )
            
            # Generic error
            raise RuntimeError(f"Error processing document with GOT-OCR: {str(e)}")
    
    @classmethod
    def release_model(cls):
        """Release the model resources."""
        # No need to do anything here since we're not loading the model directly
        # We're using subprocess calls instead
        pass

# Try to register the parser
try:
    # Only check basic imports, detailed dependency check happens in parse method
    import torch
    ParserRegistry.register(GotOcrParser)
    logger.info("GOT-OCR parser registered successfully")
except ImportError as e:
    logger.warning(f"Could not register GOT-OCR parser: {str(e)}") 