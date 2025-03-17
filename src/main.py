import parsers  # Import all parsers to ensure they're registered

from src.ui.ui import launch_ui


def main():
    launch_ui(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Explicitly disable sharing on Hugging Face
    )


if __name__ == "__main__":
    main()
