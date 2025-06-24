#!/usr/bin/env python3
"""
Simple app launcher that skips the heavy environment setup.
Use this for local development when dependencies are already installed.
"""
import sys
import os
import argparse
import shutil
from pathlib import Path

# Get the current directory and setup Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def clear_data_directories():
    """Clear all data directories (chat_history and vector_store)."""
    data_dir = Path(current_dir) / "data"
    
    directories_to_clear = [
        data_dir / "chat_history",
        data_dir / "vector_store"
    ]
    
    cleared_count = 0
    for directory in directories_to_clear:
        if directory.exists():
            try:
                # Remove all contents of the directory
                for item in directory.iterdir():
                    if item.is_file():
                        item.unlink()
                        print(f"🗑️  Removed file: {item}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        print(f"🗑️  Removed directory: {item}")
                cleared_count += len(list(directory.glob("*")))
                print(f"✅ Cleared directory: {directory}")
            except Exception as e:
                print(f"❌ Error clearing {directory}: {e}")
        else:
            print(f"ℹ️  Directory doesn't exist: {directory}")
    
    if cleared_count == 0:
        print("ℹ️  No data found to clear.")
    else:
        print(f"🎉 Successfully cleared {cleared_count} items from data directories!")

def main_with_args():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Markit v2 - Document to Markdown Converter with RAG Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_app.py                    # Run the app normally
  python run_app.py --clear-data       # Clear all data and exit
  python run_app.py --clear-data-and-run  # Clear data then run the app
        """
    )
    
    parser.add_argument(
        "--clear-data",
        action="store_true",
        help="Clear all data directories (chat_history, vector_store) and exit"
    )
    
    parser.add_argument(
        "--clear-data-and-run",
        action="store_true", 
        help="Clear all data directories then run the app"
    )
    
    args = parser.parse_args()
    
    # Handle data clearing options
    if args.clear_data or args.clear_data_and_run:
        print("🧹 Clearing data directories...")
        print("=" * 50)
        clear_data_directories()
        print("=" * 50)
        
        if args.clear_data:
            print("✅ Data clearing completed. Exiting.")
            return
        elif args.clear_data_and_run:
            print("✅ Data clearing completed. Starting app...")
            print()
    
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded environment variables from .env file")
    except ImportError:
        print("python-dotenv not installed, skipping .env file loading")
    
    # Import and run main directly
    from src.main import main
    main()

if __name__ == "__main__":
    main_with_args()