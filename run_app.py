#!/usr/bin/env python3
"""
Simple app launcher that skips the heavy environment setup.
Use this for local development when dependencies are already installed.
"""
import sys
import os

# Get the current directory and setup Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, skipping .env file loading")

# Import and run main directly
from src.main import main

if __name__ == "__main__":
    main()