import gradio as gr
import markdown
import threading
import time
import logging
from pathlib import Path
from src.core.converter import convert_file, set_cancellation_flag, is_conversion_in_progress
from src.parsers.parser_registry import ParserRegistry
from src.core.config import config
from src.core.exceptions import (
    DocumentProcessingError,
    UnsupportedFileTypeError,
    FileSizeLimitError,
    ConfigurationError
)
from src.core.logging_config import get_logger
from src.rag import rag_chat_service, document_ingestion_service

# Use centralized logging
logger = get_logger(__name__)

# Import MarkItDown to check if it's available
try:
    from markitdown import MarkItDown
    HAS_MARKITDOWN = True
    logger.info("MarkItDown is available for use")
except ImportError:
    HAS_MARKITDOWN = False
    logger.warning("MarkItDown is not available")

# Add a global variable to track cancellation state
conversion_cancelled = threading.Event()

# Pass the cancellation flag to the converter module
set_cancellation_flag(conversion_cancelled)

# Add a background thread to monitor cancellation
def monitor_cancellation():
    """Background thread to monitor cancellation and update UI if needed"""
    logger.info("Starting cancellation monitor thread")
    while is_conversion_in_progress():
        if conversion_cancelled.is_set():
            logger.info("Cancellation detected by monitor thread")
        time.sleep(0.1)  # Check every 100ms
    logger.info("Cancellation monitor thread ending")

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

def format_markdown_content(content):
    if not content:
        return content
    
    # Convert the content to HTML using markdown library
    html_content = markdown.markdown(str(content), extensions=['tables'])
    return html_content

# Function to run conversion in a separate thread
def run_conversion_thread(file_path, parser_name, ocr_method_name, output_format):
    """Run the conversion in a separate thread and return the thread object"""
    global conversion_cancelled
    
    # Reset the cancellation flag
    conversion_cancelled.clear()
    
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

def handle_convert(file_path, parser_name, ocr_method_name, output_format, is_cancelled):
    """Handle file conversion."""
    global conversion_cancelled
    
    # Check if we should cancel before starting
    if is_cancelled:
        logger.info("Conversion cancelled before starting")
        return "Conversion cancelled.", None, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    # Validate file type for the selected parser
    is_valid, error_msg = validate_file_for_parser(file_path, parser_name)
    if not is_valid:
        logger.error(f"File validation error: {error_msg}")
        return f"Error: {error_msg}", None, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    
    logger.info("Starting conversion with cancellation flag cleared")
    
    # Start the conversion in a separate thread
    thread, results = run_conversion_thread(file_path, parser_name, ocr_method_name, output_format)
    
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
    
    # Format the content and wrap it in the scrollable container
    formatted_content = format_markdown_content(str(content))
    html_output = f"<div class='output-container'>{formatted_content}</div>"
    
    logger.info("Conversion completed successfully")
    
    # Auto-ingest the converted document for RAG
    try:
        conversion_result = {
            "markdown_content": content,
            "original_filename": Path(file_path).name if file_path else "unknown",
            "conversion_method": parser_name,
            "file_size": Path(file_path).stat().st_size if file_path and Path(file_path).exists() else 0,
            "conversion_time": 0  # Could be tracked if needed
        }
        
        success, ingestion_msg, stats = document_ingestion_service.ingest_from_conversion_result(conversion_result)
        if success:
            logger.info(f"Document auto-ingested for RAG: {ingestion_msg}")
        else:
            logger.warning(f"Document ingestion failed: {ingestion_msg}")
    except Exception as e:
        logger.error(f"Error during auto-ingestion: {e}")
    
    return html_output, download_file, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def handle_chat_message(message, history):
    """Handle a new chat message with streaming response."""
    if not message or not message.strip():
        return "", history
    
    try:
        # Add user message to history
        history = history or []
        history.append({"role": "user", "content": message})
        
        # Add assistant message placeholder
        history.append({"role": "assistant", "content": ""})
        
        # Get response from RAG service
        response_text = ""
        for chunk in rag_chat_service.chat_stream(message):
            response_text += chunk
            # Update the last message in history with the current response
            history[-1]["content"] = response_text
            yield "", history
        
        logger.info(f"Chat response completed for message: {message[:50]}...")
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        if history and len(history) > 0:
            history[-1]["content"] = f"❌ {error_msg}"
        else:
            history = [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"❌ {error_msg}"}
            ]
        yield "", history

def start_new_chat_session():
    """Start a new chat session."""
    try:
        session_id = rag_chat_service.start_new_session()
        logger.info(f"Started new chat session: {session_id}")
        return [], f"✅ New chat session started: {session_id}"
    except Exception as e:
        error_msg = f"Error starting new session: {str(e)}"
        logger.error(error_msg)
        return [], f"❌ {error_msg}"

def get_chat_status():
    """Get current chat system status."""
    try:
        # Check ingestion status
        ingestion_status = document_ingestion_service.get_ingestion_status()
        
        # Check usage stats
        usage_stats = rag_chat_service.get_usage_stats()
        
        # Modern status card design with better styling
        status_html = f"""
        <div class="status-card">
            <div class="status-header">
                <h3>💬 Chat System Status</h3>
                <div class="status-indicator {'status-ready' if ingestion_status.get('system_ready', False) else 'status-not-ready'}">
                    {'🟢 READY' if ingestion_status.get('system_ready', False) else '🔴 NOT READY'}
                </div>
            </div>
            
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-label">Documents Processed</div>
                    <div class="status-value">{ingestion_status.get('processed_documents', 0)}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Vector Store</div>
                    <div class="status-value">{ingestion_status.get('total_documents_in_store', 0)} docs</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Session Usage</div>
                    <div class="status-value">{usage_stats.get('session_messages', 0)}/{usage_stats.get('session_limit', 50)}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Hourly Usage</div>
                    <div class="status-value">{usage_stats.get('hourly_messages', 0)}/{usage_stats.get('hourly_limit', 100)}</div>
                </div>
            </div>
            
            <div class="status-services">
                <div class="service-status {'service-ready' if ingestion_status.get('embedding_model_available', False) else 'service-error'}">
                    <span class="service-icon">🧠</span>
                    <span>Embedding Model</span>
                    <span class="service-indicator">{'✅' if ingestion_status.get('embedding_model_available', False) else '❌'}</span>
                </div>
                <div class="service-status {'service-ready' if ingestion_status.get('vector_store_available', False) else 'service-error'}">
                    <span class="service-icon">🗄️</span>
                    <span>Vector Store</span>
                    <span class="service-indicator">{'✅' if ingestion_status.get('vector_store_available', False) else '❌'}</span>
                </div>
            </div>
        </div>
        """
        
        return status_html
        
    except Exception as e:
        error_msg = f"Error getting chat status: {str(e)}"
        logger.error(error_msg)
        return f"""
        <div class="status-card status-error">
            <div class="status-header">
                <h3>❌ System Error</h3>
            </div>
            <p class="error-message">{error_msg}</p>
        </div>
        """

def create_ui():
    with gr.Blocks(css="""
        /* Global styles */
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Document converter styles */
        .output-container {
            max-height: 420px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
        }
        
        .gradio-container .prose {
            overflow: visible;
        }
        
        .processing-controls { 
            display: flex; 
            justify-content: center; 
            gap: 10px; 
            margin-top: 10px; 
        }
        
        .provider-options-row {
            margin-top: 15px;
            margin-bottom: 15px;
        }
        
        /* Chat Tab Styles - Complete redesign */
        .chat-tab-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .chat-header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .chat-header h2 {
            margin: 0;
            font-size: 1.8em;
            font-weight: 600;
        }
        
        .chat-header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        /* Status Card Styling */
        .status-card {
            background: #ffffff;
            border: 1px solid #e1e5e9;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .status-card:hover {
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .status-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f2f5;
        }
        
        .status-header h3 {
            margin: 0;
            color: #2c3e50;
            font-size: 1.3em;
            font-weight: 600;
        }
        
        .status-indicator {
            padding: 8px 16px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }
        
        .status-ready {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-not-ready {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .status-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        
        .status-label {
            font-size: 0.85em;
            color: #6c757d;
            margin-bottom: 5px;
            font-weight: 500;
        }
        
        .status-value {
            font-size: 1.4em;
            font-weight: 700;
            color: #495057;
        }
        
        .status-services {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .service-status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 15px;
            border-radius: 8px;
            font-weight: 500;
            flex: 1;
            min-width: 200px;
        }
        
        .service-ready {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .service-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .service-icon {
            font-size: 1.2em;
        }
        
        .service-indicator {
            margin-left: auto;
        }
        
        .status-error {
            border-color: #dc3545;
            background: #f8d7da;
        }
        
        .error-message {
            color: #721c24;
            margin: 0;
            font-weight: 500;
        }
        
        /* Control buttons styling */
        .control-buttons {
            display: flex;
            gap: 12px;
            justify-content: flex-end;
            margin-bottom: 25px;
        }
        
        .control-btn {
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
        }
        
        .btn-refresh {
            background: #17a2b8;
            color: white;
        }
        
        .btn-refresh:hover {
            background: #138496;
            transform: translateY(-1px);
        }
        
        .btn-new-session {
            background: #28a745;
            color: white;
        }
        
        .btn-new-session:hover {
            background: #218838;
            transform: translateY(-1px);
        }
        
        /* Chat interface styling */
        .chat-main-container {
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            overflow: hidden;
            margin-bottom: 25px;
        }
        
        .chat-container {
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #e1e5e9;
            overflow: hidden;
        }
        
        /* Custom chatbot styling */
        .gradio-chatbot {
            border: none !important;
            background: #ffffff;
        }
        
        .gradio-chatbot .message {
            padding: 15px 20px;
            margin: 10px;
            border-radius: 12px;
        }
        
        .gradio-chatbot .message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 50px;
        }
        
        .gradio-chatbot .message.assistant {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            margin-right: 50px;
        }
        
        /* Input area styling */
        .chat-input-container {
            background: #ffffff;
            padding: 20px;
            border-top: 1px solid #e1e5e9;
            border-radius: 0 0 15px 15px;
        }
        
        .input-row {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        
        .message-input {
            flex: 1;
            border: 2px solid #e1e5e9;
            border-radius: 25px;
            padding: 12px 20px;
            font-size: 1em;
            transition: all 0.3s ease;
            resize: none;
            max-height: 120px;
            min-height: 48px;
        }
        
        .message-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            outline: none;
        }
        
        .send-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            min-width: 80px;
            height: 48px;
            margin-right: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1em;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        .send-button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        /* Session info styling */
        .session-info {
            background: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            padding: 15px;
            color: #0056b3;
            font-weight: 500;
            text-align: center;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .chat-tab-container {
                padding: 10px;
            }
            
            .status-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .service-status {
                min-width: 100%;
            }
            
            .control-buttons {
                flex-direction: column;
                gap: 8px;
            }
            
            .gradio-chatbot .message.user {
                margin-left: 20px;
            }
            
            .gradio-chatbot .message.assistant {
                margin-right: 20px;
            }
        }
    """) as demo:
        # Modern title with better styling
        gr.Markdown("""
        # 🚀 Markit
        ## Document to Markdown Converter with RAG Chat
        """)
        
        with gr.Tabs():
            # Document Converter Tab
            with gr.TabItem("📄 Document Converter"):
                # State to track if cancellation is requested
                cancel_requested = gr.State(False)
                # State to store the conversion thread
                conversion_thread = gr.State(None)
                # State to store the output format (fixed to Markdown)
                output_format_state = gr.State("Markdown")

                # File input first
                file_input = gr.File(label="Upload Document", type="filepath")
                
                # Provider and OCR options below the file input
                with gr.Row(elem_classes=["provider-options-row"]):
                    with gr.Column(scale=1):
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
                
                # Simple output container with just one scrollbar
                file_display = gr.HTML(
                    value="<div class='output-container'></div>",
                    label="Converted Content"
                )
                
                file_download = gr.File(label="Download File")
                
                # Processing controls row
                with gr.Row(elem_classes=["processing-controls"]):
                    convert_button = gr.Button("Convert", variant="primary")
                    cancel_button = gr.Button("Cancel", variant="stop", visible=False)

                # Event handlers for document converter
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
                    global conversion_cancelled
                    conversion_cancelled.clear()
                    logger.info("Starting conversion with cancellation flag cleared")
                    return gr.update(visible=False), gr.update(visible=True), False

                # Set cancel flag and terminate thread when cancel button is clicked
                def request_cancellation(thread):
                    global conversion_cancelled
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
                    inputs=[file_input, provider_dropdown, ocr_dropdown, output_format_state, cancel_requested],
                    outputs=[file_display, file_download, convert_button, cancel_button, conversion_thread]
                )
                
                # Handle cancel button click
                cancel_button.click(
                    fn=request_cancellation,
                    inputs=[conversion_thread],
                    outputs=[convert_button, cancel_button, cancel_requested, conversion_thread],
                    queue=False  # Execute immediately
                )

            # Chat Tab - Completely redesigned
            with gr.TabItem("💬 Chat with Documents"):
                with gr.Column(elem_classes=["chat-tab-container"]):
                    # Modern header
                    gr.HTML("""
                    <div class="chat-header">
                        <h2>💬 Chat with your converted documents</h2>
                        <p>Ask questions about your documents using advanced RAG technology</p>
                    </div>
                    """)
                    
                    # Status section with modern design
                    status_display = gr.HTML(value=get_chat_status())
                    
                    # Control buttons
                    with gr.Row(elem_classes=["control-buttons"]):
                        refresh_status_btn = gr.Button("🔄 Refresh Status", elem_classes=["control-btn", "btn-refresh"])
                        new_session_btn = gr.Button("🆕 New Session", elem_classes=["control-btn", "btn-new-session"])
                    
                    # Main chat interface
                    with gr.Column(elem_classes=["chat-main-container"]):
                        chatbot = gr.Chatbot(
                            elem_classes=["chat-container"],
                            height=500,
                            show_label=False,
                            show_share_button=False,
                            bubble_full_width=False,
                            type="messages",
                            placeholder="Start a conversation by asking questions about your documents..."
                        )
                        
                        # Input area
                        with gr.Row(elem_classes=["input-row"]):
                            msg_input = gr.Textbox(
                                placeholder="Ask questions about your documents...",
                                show_label=False,
                                scale=5,
                                lines=1,
                                max_lines=3,
                                elem_classes=["message-input"]
                            )
                            send_btn = gr.Button("Submit", elem_classes=["send-button"], scale=0)
                    
                    # Session info with better styling
                    session_info = gr.HTML(
                        value='<div class="session-info">No active session - Click "New Session" to start</div>'
                    )
                
                # Event handlers for chat
                def clear_input():
                    return ""
                
                # Send message when button clicked or Enter pressed
                msg_input.submit(
                    fn=handle_chat_message,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot]
                )
                
                send_btn.click(
                    fn=handle_chat_message,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot]
                )
                
                # New session handler with improved feedback
                def enhanced_new_session():
                    history, info = start_new_chat_session()
                    session_html = f'<div class="session-info">{info}</div>'
                    return history, session_html
                
                new_session_btn.click(
                    fn=enhanced_new_session,
                    inputs=[],
                    outputs=[chatbot, session_info]
                )
                
                # Refresh status handler
                refresh_status_btn.click(
                    fn=get_chat_status,
                    inputs=[],
                    outputs=[status_display]
                )

    return demo


def launch_ui(server_name="0.0.0.0", server_port=7860, share=False):
    demo = create_ui()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        root_path="",
        show_error=True,
        share=share
    ) 