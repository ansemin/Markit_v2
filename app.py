import spaces  # Must be imported before any CUDA initialization
import sys
import os
from pathlib import Path

# Get the current directory and setup Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import environment manager after setting up path
try:
    from src.core.environment import environment_manager
    
    # Perform complete environment setup
    print("Setting up environment...")
    setup_results = environment_manager.full_environment_setup()
    
    # Report setup status
    print(f"Environment setup completed with results: {len([k for k, v in setup_results.items() if v])} successful, {len([k for k, v in setup_results.items() if not v])} failed")
    
except ImportError as e:
    print(f"Warning: Could not import environment manager: {e}")
    print("Falling back to basic setup...")
    
    # Fallback to basic setup if environment manager fails
    import subprocess
    
    # Basic dependency checks
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("Installing PyTorch...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"], check=False)
    
    try:
        from markitdown import MarkItDown
        print("MarkItDown is available")
    except ImportError:
        print("Installing MarkItDown...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "markitdown[all]"], check=False)

# Import main function with fallback strategies (HF Spaces compatibility)
try:
    from src.main import main
except ModuleNotFoundError:
    try:
        # Fallback: adjust path and try again
        sys.path.append(os.path.join(current_dir, "src"))
        from src.main import main
    except ModuleNotFoundError:
        # Last resort: create __init__.py if missing
        init_path = os.path.join(current_dir, "src", "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                pass
        from src.main import main

if __name__ == "__main__":
    main()