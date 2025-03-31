import spaces  # Must be imported before any CUDA initialization
import sys
import os
import subprocess
import shutil
from pathlib import Path
import logging

# Configure logging - Add this section to suppress httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)  # Raise level to WARNING to suppress INFO logs
logging.getLogger("urllib3").setLevel(logging.WARNING)  # Also suppress urllib3 logs which might be used
logging.getLogger("httpcore").setLevel(logging.WARNING)  # httpcore is used by httpx

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Run setup.sh at startup
try:
    setup_script = os.path.join(current_dir, "setup.sh")
    if os.path.exists(setup_script):
        print("Running setup.sh...")
        subprocess.run(["bash", setup_script], check=False)
        print("setup.sh completed")
except Exception as e:
    print(f"Error running setup.sh: {e}")

# Check if spaces module is installed (needed for ZeroGPU)
try:
    print("Spaces module found for ZeroGPU support")
except ImportError:
    print("WARNING: Spaces module not found. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spaces"], check=False)

# Check for PyTorch and CUDA availability (needed for GOT-OCR)
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available. GOT-OCR performs best with GPU acceleration.")
except ImportError:
    print("WARNING: PyTorch not installed. Installing PyTorch...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"], check=False)

# Check if transformers is installed (needed for GOT-OCR)
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("WARNING: Transformers not installed. Installing transformers from GitHub...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/huggingface/transformers.git@main", "accelerate", "verovio"], check=False)

# Check if numpy is installed with the correct version
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    if np.__version__ != "1.26.3":
        print("WARNING: NumPy version mismatch. Installing exact version 1.26.3...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.3"], check=False)
except ImportError:
    print("WARNING: NumPy not installed. Installing NumPy 1.26.3...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.3"], check=False)

# Check if markitdown is installed
try:
    from markitdown import MarkItDown
    print("MarkItDown is installed")
except ImportError:
    print("WARNING: MarkItDown not installed. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "markitdown[all]"], check=False)
    try:
        from markitdown import MarkItDown
        print("MarkItDown installed successfully")
    except ImportError:
        print("ERROR: Failed to install MarkItDown")

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, skipping .env file loading")

# Load API keys from environment variables
gemini_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if API keys are available and print messages
if not gemini_api_key:
    print("Warning: GOOGLE_API_KEY environment variable not found. Gemini Flash parser and LaTeX to Markdown conversion may not work.")
else:
    print(f"Found Gemini API key: {gemini_api_key[:5]}...{gemini_api_key[-5:] if len(gemini_api_key) > 10 else ''}")
    print("Gemini API will be used for LaTeX to Markdown conversion when using GOT-OCR with Formatted Text mode")

if not openai_api_key:
    print("Warning: OPENAI_API_KEY environment variable not found. LLM-based image description in MarkItDown may not work.")
else:
    print(f"Found OpenAI API key: {openai_api_key[:5]}...{openai_api_key[-5:] if len(openai_api_key) > 10 else ''}")
    print("OpenAI API will be available for LLM-based image descriptions in MarkItDown")

# Add the current directory to the Python path
sys.path.append(current_dir)

# Try different import approaches
try:
    # First attempt - standard import
    from src.main import main
except ModuleNotFoundError:
    try:
        # Second attempt - adjust path and try again
        sys.path.append(os.path.join(current_dir, "src"))
        from src.main import main
    except ModuleNotFoundError:
        # Third attempt - create __init__.py if it doesn't exist
        init_path = os.path.join(current_dir, "src", "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                pass  # Create empty __init__.py file
        # Try import again
        from src.main import main

if __name__ == "__main__":
    main()