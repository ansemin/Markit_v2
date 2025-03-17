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

# Function to setup Tesseract
def setup_tesseract():
    """Setup Tesseract OCR environment."""
    print("Setting up Tesseract OCR environment...")
    
    # Create tessdata directory if it doesn't exist
    tessdata_dir = os.path.join(current_dir, "tessdata")
    os.makedirs(tessdata_dir, exist_ok=True)
    
    # Set TESSDATA_PREFIX environment variable if not already set
    if not os.environ.get('TESSDATA_PREFIX'):
        # Check multiple possible locations
        possible_tessdata_dirs = [
            tessdata_dir,  # Our local tessdata directory
            "/usr/share/tesseract-ocr/4.00/tessdata",  # Common location in Hugging Face
            "/usr/share/tesseract-ocr/tessdata",  # Another common location
            "/usr/local/share/tessdata",  # Standard installation location
        ]
        
        # Use the first directory that exists
        for dir_path in possible_tessdata_dirs:
            if os.path.exists(dir_path):
                os.environ['TESSDATA_PREFIX'] = dir_path
                print(f"Set TESSDATA_PREFIX to {dir_path}")
                break
        else:
            # If none exist, use our local directory
            os.environ['TESSDATA_PREFIX'] = tessdata_dir
            print(f"No existing tessdata directory found, set TESSDATA_PREFIX to {tessdata_dir}")
    
    # Download eng.traineddata if it doesn't exist in our local tessdata
    eng_traineddata = os.path.join(tessdata_dir, "eng.traineddata")
    if not os.path.exists(eng_traineddata):
        try:
            print("Downloading eng.traineddata...")
            url = "https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata"
            urllib.request.urlretrieve(url, eng_traineddata)
            print("Downloaded eng.traineddata")
        except Exception as e:
            print(f"Error downloading eng.traineddata: {e}")
    
    # Configure pytesseract
    try:
        import pytesseract
        # Check if tesseract is in PATH
        tesseract_cmd = shutil.which("tesseract")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            print(f"Set pytesseract.tesseract_cmd to {tesseract_cmd}")
        else:
            # Try common locations
            common_locations = [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                "/app/tesseract/tesseract"
            ]
            for location in common_locations:
                if os.path.isfile(location) and os.access(location, os.X_OK):
                    pytesseract.pytesseract.tesseract_cmd = location
                    print(f"Set pytesseract.tesseract_cmd to {location}")
                    break
            else:
                print("Warning: Could not find tesseract executable")
    except ImportError:
        print("pytesseract not installed")
    
    # Try to import tesserocr to verify it's working
    try:
        import tesserocr
        print(f"tesserocr imported successfully, version: {tesserocr.tesseract_version()}")
    except ImportError:
        print("tesserocr not installed or not working")
    except Exception as e:
        print(f"Error importing tesserocr: {e}")

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

# Call setup function at import time
setup_tesseract()

if __name__ == "__main__":
    main()