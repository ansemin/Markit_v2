"""
Environment setup and dependency management for the Markit application.
Extracted from app.py to improve code organization while maintaining HF Spaces compatibility.
"""
import os
import sys
import subprocess
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

from src.core.config import config
from src.core.logging_config import setup_logging


class EnvironmentManager:
    """Manages environment setup and dependency installation."""
    
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.logger = logging.getLogger(__name__)
    
    def run_setup_script(self) -> bool:
        """Run setup.sh script if it exists."""
        try:
            setup_script = os.path.join(self.current_dir, "setup.sh")
            if os.path.exists(setup_script):
                print("Running setup.sh...")
                subprocess.run(["bash", setup_script], check=False)
                print("setup.sh completed")
                return True
        except Exception as e:
            print(f"Error running setup.sh: {e}")
        return False
    
    def check_spaces_module(self) -> bool:
        """Check and install spaces module for ZeroGPU support."""
        try:
            import spaces
            print("Spaces module found for ZeroGPU support")
            return True
        except ImportError:
            print("WARNING: Spaces module not found. Installing...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spaces"], check=False)
                return True
            except Exception as e:
                print(f"Error installing spaces module: {e}")
                return False
    
    def check_pytorch(self) -> Tuple[bool, Dict[str, str]]:
        """Check PyTorch and CUDA availability."""
        info = {}
        try:
            import torch
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = str(torch.cuda.is_available())
            
            print(f"PyTorch version: {info['pytorch_version']}")
            print(f"CUDA available: {info['cuda_available']}")
            
            if torch.cuda.is_available():
                info["cuda_device"] = torch.cuda.get_device_name(0)
                info["cuda_version"] = torch.version.cuda
                print(f"CUDA device: {info['cuda_device']}")
                print(f"CUDA version: {info['cuda_version']}")
            else:
                print("WARNING: CUDA not available. GOT-OCR performs best with GPU acceleration.")
            
            return True, info
        except ImportError:
            print("WARNING: PyTorch not installed. Installing PyTorch...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"], check=False)
                return True, info
            except Exception as e:
                print(f"Error installing PyTorch: {e}")
                return False, info
    
    def check_transformers(self) -> bool:
        """Check and install transformers library."""
        try:
            import transformers
            print(f"Transformers version: {transformers.__version__}")
            return True
        except ImportError:
            print("WARNING: Transformers not installed. Installing transformers from GitHub...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-q", 
                    "git+https://github.com/huggingface/transformers.git@main", 
                    "accelerate", "verovio"
                ], check=False)
                return True
            except Exception as e:
                print(f"Error installing transformers: {e}")
                return False
    
    def check_numpy(self) -> bool:
        """Check and install correct NumPy version."""
        try:
            import numpy as np
            print(f"NumPy version: {np.__version__}")
            if np.__version__ != "1.26.3":
                print("WARNING: NumPy version mismatch. Installing exact version 1.26.3...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.3"], check=False)
            return True
        except ImportError:
            print("WARNING: NumPy not installed. Installing NumPy 1.26.3...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.3"], check=False)
                return True
            except Exception as e:
                print(f"Error installing NumPy: {e}")
                return False
    
    def check_markitdown(self) -> bool:
        """Check and install MarkItDown library."""
        try:
            from markitdown import MarkItDown
            print("MarkItDown is installed")
            return True
        except ImportError:
            print("WARNING: MarkItDown not installed. Installing...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "markitdown[all]"], check=False)
                from markitdown import MarkItDown
                print("MarkItDown installed successfully")
                return True
            except ImportError:
                print("ERROR: Failed to install MarkItDown")
                return False
            except Exception as e:
                print(f"Error installing MarkItDown: {e}")
                return False
    
    def load_environment_variables(self) -> bool:
        """Load environment variables from .env file."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("Loaded environment variables from .env file")
            return True
        except ImportError:
            print("python-dotenv not installed, skipping .env file loading")
            return False
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate and report API key availability."""
        results = {}
        
        # Check Gemini API key
        gemini_key = config.api.google_api_key
        if not gemini_key:
            print("Warning: GOOGLE_API_KEY environment variable not found. Gemini Flash parser and LaTeX to Markdown conversion may not work.")
            results["gemini"] = False
        else:
            print(f"Found Gemini API key: {gemini_key[:5]}...{gemini_key[-5:] if len(gemini_key) > 10 else ''}")
            print("Gemini API will be used for LaTeX to Markdown conversion when using GOT-OCR with Formatted Text mode")
            results["gemini"] = True
        
        # Check OpenAI API key
        openai_key = config.api.openai_api_key
        if not openai_key:
            print("Warning: OPENAI_API_KEY environment variable not found. LLM-based image description in MarkItDown may not work.")
            results["openai"] = False
        else:
            print(f"Found OpenAI API key: {openai_key[:5]}...{openai_key[-5:] if len(openai_key) > 10 else ''}")
            print("OpenAI API will be available for LLM-based image descriptions in MarkItDown")
            results["openai"] = True
        
        # Check Mistral API key
        mistral_key = config.api.mistral_api_key
        if mistral_key:
            print(f"Found Mistral API key: {mistral_key[:5]}...{mistral_key[-5:] if len(mistral_key) > 10 else ''}")
            results["mistral"] = True
        else:
            results["mistral"] = False
        
        return results
    
    def setup_python_path(self) -> None:
        """Setup Python path for imports."""
        if self.current_dir not in sys.path:
            sys.path.append(self.current_dir)
    
    def setup_logging(self) -> None:
        """Setup centralized logging configuration."""
        # Configure logging to suppress httpx and other noisy logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        # Setup our centralized logging
        setup_logging()
    
    def full_environment_setup(self) -> Dict[str, bool]:
        """
        Perform complete environment setup.
        
        Returns:
            Dictionary with setup results for each component
        """
        results = {}
        
        # Setup logging first
        self.setup_logging()
        
        # Run setup script
        results["setup_script"] = self.run_setup_script()
        
        # Check and install dependencies
        results["spaces_module"] = self.check_spaces_module()
        results["pytorch"], pytorch_info = self.check_pytorch()
        results["transformers"] = self.check_transformers()
        results["numpy"] = self.check_numpy()
        results["markitdown"] = self.check_markitdown()
        
        # Load environment variables
        results["env_vars"] = self.load_environment_variables()
        
        # Validate API keys
        api_keys = self.validate_api_keys()
        results["api_keys"] = api_keys
        
        # Setup Python path
        self.setup_python_path()
        results["python_path"] = True
        
        # Validate configuration
        validation = config.validate()
        results["config_valid"] = validation["valid"]
        
        if validation["warnings"]:
            for warning in validation["warnings"]:
                print(f"Configuration warning: {warning}")
        
        if validation["errors"]:
            for error in validation["errors"]:
                print(f"Configuration error: {error}")
        
        return results


# Global instance
environment_manager = EnvironmentManager()