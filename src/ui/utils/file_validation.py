"""File validation utilities for the UI components."""

import gradio as gr
import logging
from pathlib import Path

from src.core.config import config
from src.core.logging_config import get_logger

logger = get_logger(__name__)


def update_ui_for_file_count(files):
    """Update UI components based on the number of files uploaded."""
    if not files or len(files) == 0:
        return (
            gr.update(visible=False),  # processing_type_selector
            "<div style='color: #666; font-style: italic;'>Upload documents to begin</div>"  # file_status_text
        )
    
    if len(files) == 1:
        file_name = files[0].name if hasattr(files[0], 'name') else str(files[0])
        return (
            gr.update(visible=False),  # processing_type_selector (hidden for single file)
            f"<div style='color: #2563eb; font-weight: 500;'>📄 Single document: {file_name}</div>"
        )
    else:
        # Calculate total size for validation display
        total_size = 0
        try:
            for file in files:
                if hasattr(file, 'size'):
                    total_size += file.size
                elif hasattr(file, 'name'):
                    # For file paths, get size from filesystem
                    total_size += Path(file.name).stat().st_size
        except:
            pass  # Size calculation is optional for display
        
        size_display = f" ({total_size / (1024*1024):.1f}MB)" if total_size > 0 else ""
        
        # Check if within limits
        if len(files) > 5:
            status_color = "#dc2626"  # red
            status_text = f"⚠️ Too many files: {len(files)}/5 (max 5 files allowed)"
        elif total_size > 20 * 1024 * 1024:  # 20MB
            status_color = "#dc2626"  # red
            status_text = f"⚠️ Files too large{size_display} (max 20MB combined)"
        else:
            status_color = "#059669"  # green
            status_text = f"📂 Batch mode: {len(files)} files{size_display}"
        
        return (
            gr.update(visible=True),  # processing_type_selector (visible for multiple files)
            f"<div style='color: {status_color}; font-weight: 500;'>{status_text}</div>"
        )


def validate_file_for_parser(file_path, parser_name):
    """Validate if the file type is supported by the selected parser."""
    if not file_path:
        return True, ""  # No file selected yet
    
    try:
        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()
        
        # Check file size
        if file_path_obj.exists():
            file_size = file_path_obj.stat().st_size
            if file_size > config.app.max_file_size:
                size_mb = file_size / (1024 * 1024)
                max_mb = config.app.max_file_size / (1024 * 1024)
                return False, f"File size ({size_mb:.1f}MB) exceeds maximum allowed size ({max_mb:.1f}MB)"
        
        # Check file extension
        if file_ext not in config.app.allowed_extensions:
            return False, f"File type '{file_ext}' is not supported. Allowed types: {', '.join(config.app.allowed_extensions)}"
        
        # Parser-specific validation
        if "GOT-OCR" in parser_name:
            if file_ext not in ['.jpg', '.jpeg', '.png']:
                return False, "GOT-OCR only supports JPG and PNG formats."
        
        return True, ""
        
    except Exception as e:
        logger.error(f"Error validating file: {e}")
        return False, f"Error validating file: {e}"