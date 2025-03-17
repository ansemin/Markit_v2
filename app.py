import sys
import os
import subprocess
import shutil
from pathlib import Path
import urllib.request

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

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, skipping .env file loading")

# Load Gemini API key from environment variable
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key is available and print a message if not
if not gemini_api_key:
    print("Warning: GOOGLE_API_KEY environment variable not found. Gemini Flash parser may not work.")
else:
    print(f"Found Gemini API key: {gemini_api_key[:5]}...{gemini_api_key[-5:] if len(gemini_api_key) > 10 else ''}")

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