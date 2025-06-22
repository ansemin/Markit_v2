import os
from src import parsers  # Import all parsers to ensure they're registered
from src.ui.ui import launch_ui

def main():
    # Detect if running in Hugging Face Spaces
    is_hf_space = os.getenv("SPACE_ID") is not None
    
    if is_hf_space:
        # Hugging Face Spaces configuration
        launch_ui(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    else:
        # Local development configuration
        launch_ui(
            server_name="localhost",
            server_port=7860,
            share=False
        )


if __name__ == "__main__":
    main()
