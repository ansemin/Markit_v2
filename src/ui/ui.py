"""Main UI orchestrator - Refactored modular interface for Markit application."""

import gradio as gr
import logging

from src.core.converter import set_cancellation_flag
from src.core.logging_config import get_logger
from src.ui.styles.ui_styles import CSS_STYLES
from src.ui.components.document_converter import create_document_converter_tab
from src.ui.components.chat_interface import create_chat_interface_tab
from src.ui.components.query_ranker import create_query_ranker_tab
from src.ui.utils.threading_utils import get_cancellation_event

logger = get_logger(__name__)

# Import MarkItDown to check if it's available
try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
    logger.info("MarkItDown is available for use")
except ImportError:
    HAS_MARKITDOWN = False
    logger.warning("MarkItDown is not available")

# Initialize global cancellation event and pass to converter module
conversion_cancelled = get_cancellation_event()
set_cancellation_flag(conversion_cancelled)


def create_ui():
    """Create the main Gradio interface with all tabs."""
    with gr.Blocks(css=CSS_STYLES) as demo:
        # Modern title with better styling
        gr.Markdown("""
        # 🚀 Markit
        ## Document to Markdown Converter with RAG Chat
        """)
        
        with gr.Tabs():
            # Create all tabs using component functions
            create_document_converter_tab()
            create_chat_interface_tab()
            create_query_ranker_tab()
    
    return demo


def launch_ui(share=False, server_name="0.0.0.0", server_port=7860):
    """Launch the Gradio interface."""
    logger.info("Creating and launching UI...")
    demo = create_ui()
    return demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True
    )