"""Document converter UI component and logic."""

import threading
import time
import gradio as gr
import logging
from pathlib import Path

from src.core.converter import convert_file
from src.core.logging_config import get_logger
from src.services.document_service import DocumentService
from src.rag import document_ingestion_service
from src.ui.utils.file_validation import validate_file_for_parser
from src.ui.utils.threading_utils import (
    conversion_cancelled, 
    monitor_cancellation, 
    reset_cancellation,
    set_cancellation
)
from src.ui.formatters.content_formatters import format_markdown_content, format_latex_content

logger = get_logger(__name__)


def run_conversion_thread(file_path, parser_name, ocr_method_name, output_format):
    """Run the conversion in a separate thread and return the thread object"""
    # Reset the cancellation flag
    reset_cancellation()
    
    # Create a container for the results
    results = {"content": None, "download_file": None, "error": None}
    
    def conversion_worker():
        try:
            content, download_file = convert_file(file_path, parser_name, ocr_method_name, output_format)
            results["content"] = content
            results["download_file"] = download_file
        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            results["error"] = str(e)
    
    # Create and start the thread
    thread = threading.Thread(target=conversion_worker)
    thread.daemon = True
    thread.start()
    
    return thread, results


def run_conversion_thread_multi(file_paths, parser_name, ocr_method_name, output_format, processing_type):
    """Run the conversion in a separate thread for multiple files."""
    # Results will be shared between threads
    results = {"content": None, "download_file": None, "error": None}
    
    def conversion_worker():
        try:
            logger.info(f"Starting multi-file conversion thread for {len(file_paths)} files")
            
            # Use the new document service unified method
            document_service = DocumentService()
            document_service.set_cancellation_flag(conversion_cancelled)
            
            # Call the unified convert_documents method
            content, output_file = document_service.convert_documents(
                file_paths=file_paths,
                parser_name=parser_name,
                ocr_method_name=ocr_method_name,
                output_format=output_format,
                processing_type=processing_type
            )
            
            logger.info(f"Multi-file conversion completed successfully for {len(file_paths)} files")
            results["content"] = content
            results["download_file"] = output_file
            
        except Exception as e:
            logger.error(f"Error during multi-file conversion: {str(e)}")
            results["error"] = str(e)
    
    # Create and start the thread
    thread = threading.Thread(target=conversion_worker)
    thread.daemon = True
    thread.start()
    
    return thread, results


def handle_convert(files, parser_name, ocr_method_name, output_format, processing_type, is_cancelled):
    """Handle file conversion for single or multiple files."""
    # Check if we should cancel before starting
    if is_cancelled:
        logger.info("Conversion cancelled before starting")
        return "Conversion cancelled.", None, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    # Validate files input
    if not files or len(files) == 0:
        error_msg = "No files uploaded. Please upload at least one document."
        logger.error(error_msg)
        return f"Error: {error_msg}", None, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    # Convert Gradio file objects to file paths
    file_paths = []
    for file in files:
        if hasattr(file, 'name'):
            file_paths.append(file.name)
        else:
            file_paths.append(str(file))
    
    # Validate file types for the selected parser
    for file_path in file_paths:
        is_valid, error_msg = validate_file_for_parser(file_path, parser_name)
        if not is_valid:
            logger.error(f"File validation error: {error_msg}")
            return f"Error: {error_msg}", None, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    logger.info(f"Starting conversion of {len(file_paths)} file(s) with cancellation flag cleared")
    
    # Start the conversion in a separate thread
    thread, results = run_conversion_thread_multi(file_paths, parser_name, ocr_method_name, output_format, processing_type)
    
    # Start the monitoring thread
    monitor_thread = threading.Thread(target=monitor_cancellation)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Wait for the thread to complete or be cancelled
    while thread.is_alive():
        # Check if cancellation was requested
        if conversion_cancelled.is_set():
            logger.info("Cancellation detected, waiting for thread to finish")
            # Give the thread a chance to clean up
            thread.join(timeout=0.5)
            if thread.is_alive():
                logger.warning("Thread did not finish within timeout")
            return "Conversion cancelled.", None, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        # Sleep briefly to avoid busy waiting
        time.sleep(0.1)
    
    # Thread has completed, check results
    if results["error"]:
        return f"Error: {results['error']}", None, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    content = results["content"]
    download_file = results["download_file"]
    
    # If conversion returned a cancellation message
    if content == "Conversion cancelled.":
        logger.info("Converter returned cancellation message")
        return content, None, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    # Format the content based on parser type
    if "GOT-OCR" in parser_name:
        # For GOT-OCR, display as LaTeX
        formatted_content = format_latex_content(str(content))
        html_output = f"<div class='output-container'>{formatted_content}</div>"
    else:
        # For other parsers, display as Markdown
        formatted_content = format_markdown_content(str(content))
        html_output = f"<div class='output-container'>{formatted_content}</div>"
    
    logger.info("Conversion completed successfully")
    
    # Auto-ingest the converted document for RAG
    try:
        # For multi-file conversion, use the first file for metadata
        file_path = file_paths[0] if file_paths else None
        
        # Read original file content for proper deduplication hashing
        original_file_content = None
        if file_path and Path(file_path).exists():
            try:
                with open(file_path, 'rb') as f:
                    original_file_content = f.read().decode('utf-8', errors='ignore')
            except Exception as e:
                logger.warning(f"Could not read original file content: {e}")
        
        conversion_result = {
            "markdown_content": content,
            "original_filename": Path(file_path).name if file_path else "unknown",
            "conversion_method": parser_name,
            "file_size": Path(file_path).stat().st_size if file_path and Path(file_path).exists() else 0,
            "conversion_time": 0,  # Could be tracked if needed
            "original_file_content": original_file_content
        }
        
        success, ingestion_msg, stats = document_ingestion_service.ingest_from_conversion_result(conversion_result)
        if success:
            logger.info(f"Document auto-ingested for RAG: {ingestion_msg}")
        else:
            logger.warning(f"Document ingestion failed: {ingestion_msg}")
    except Exception as e:
        logger.error(f"Error during auto-ingestion: {e}")
    
    return html_output, download_file, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)




def create_document_converter_tab():
    """Create the document converter tab UI."""
    with gr.TabItem("📄 Document Converter"):
        with gr.Column(elem_classes=["chat-tab-container"]):
            # Modern header matching other tabs
            gr.HTML("""
            <div class="chat-header">
                <h2>📄 Document Converter</h2>
                <p>Convert documents to Markdown format with advanced OCR and AI processing</p>
            </div>
            """)
            
            # State to track if cancellation is requested
            cancel_requested = gr.State(False)
            # State to store the conversion thread
            conversion_thread = gr.State(None)
            # State to store the output format (fixed to Markdown)
            output_format_state = gr.State("Markdown")

            # Multi-file input (supports single and multiple files)
            files_input = gr.Files(
                label="Upload Document(s) - Single file or up to 5 files (20MB max combined)",
                file_count="multiple",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".txt", ".md", ".html", ".htm", ".csv"]
            )
            
            # Processing type selector (visible only for multiple files)
            processing_type_selector = gr.Radio(
                choices=["combined", "individual", "summary", "comparison"],
                value="combined",
                label="Multi-Document Processing Type",
                info="How to process multiple documents together",
                visible=False
            )
            
            # Status text to show file count and processing mode
            file_status_text = gr.HTML(
                value="<div style='color: #666; font-style: italic;'>Upload documents to begin</div>",
                label=""
            )
            
            # Provider and OCR options below the file input
            with gr.Row(elem_classes=["provider-options-row"]):
                with gr.Column(scale=1):
                    from src.parsers.parser_registry import ParserRegistry
                    parser_names = ParserRegistry.get_parser_names()
                    
                    # Make MarkItDown the default parser if available
                    default_parser = next((p for p in parser_names if p == "MarkItDown"), parser_names[0] if parser_names else "PyPdfium")
                    
                    provider_dropdown = gr.Dropdown(
                        label="Provider",
                        choices=parser_names,
                        value=default_parser,
                        interactive=True
                    )
                with gr.Column(scale=1):
                    default_ocr_options = ParserRegistry.get_ocr_options(default_parser)
                    default_ocr = default_ocr_options[0] if default_ocr_options else "No OCR"
                    
                    ocr_dropdown = gr.Dropdown(
                        label="OCR Options",
                        choices=default_ocr_options,
                        value=default_ocr,
                        interactive=True
                    )
            
            # Processing controls row with consistent styling
            with gr.Row(elem_classes=["control-buttons"]):
                convert_button = gr.Button("🚀 Convert", elem_classes=["control-btn", "btn-primary"])
                cancel_button = gr.Button("⏹️ Cancel", elem_classes=["control-btn", "btn-clear-data"], visible=False)
            
            # Simple output container with just one scrollbar
            file_display = gr.HTML(
                value="<div class='output-container'></div>",
                label="Converted Content"
            )
            
            file_download = gr.File(label="Download File")
            
            # Event handlers
            from src.ui.utils.file_validation import update_ui_for_file_count
            
            # Update UI when files are uploaded
            files_input.change(
                fn=update_ui_for_file_count,
                inputs=[files_input],
                outputs=[processing_type_selector, file_status_text]
            )
            
            provider_dropdown.change(
                lambda p: gr.Dropdown(
                    choices=["Plain Text", "Formatted Text"] if "GOT-OCR" in p else ParserRegistry.get_ocr_options(p),
                    value="Plain Text" if "GOT-OCR" in p else (ParserRegistry.get_ocr_options(p)[0] if ParserRegistry.get_ocr_options(p) else None)
                ),
                inputs=[provider_dropdown],
                outputs=[ocr_dropdown]
            )
            
            # Reset cancel flag when starting conversion
            def start_conversion():
                from src.ui.utils.threading_utils import conversion_cancelled
                conversion_cancelled.clear()
                logger.info("Starting conversion with cancellation flag cleared")
                return gr.update(visible=False), gr.update(visible=True), False

            # Set cancel flag and terminate thread when cancel button is clicked
            def request_cancellation(thread):
                from src.ui.utils.threading_utils import conversion_cancelled
                conversion_cancelled.set()
                logger.info("Cancel button clicked, cancellation flag set")
                
                # Try to join the thread with a timeout
                if thread is not None:
                    logger.info(f"Attempting to join conversion thread: {thread}")
                    thread.join(timeout=0.5)
                    if thread.is_alive():
                        logger.warning("Thread did not finish within timeout")
                
                # Add immediate feedback to the user
                return gr.update(visible=True), gr.update(visible=False), True, None
            
            # Start conversion sequence
            convert_button.click(
                fn=start_conversion,
                inputs=[],
                outputs=[convert_button, cancel_button, cancel_requested],
                queue=False  # Execute immediately
            ).then(
                fn=handle_convert,
                inputs=[files_input, provider_dropdown, ocr_dropdown, output_format_state, processing_type_selector, cancel_requested],
                outputs=[file_display, file_download, convert_button, cancel_button, conversion_thread]
            )
            
            # Handle cancel button click
            cancel_button.click(
                fn=request_cancellation,
                inputs=[conversion_thread],
                outputs=[convert_button, cancel_button, cancel_requested, conversion_thread],
                queue=False  # Execute immediately
            )