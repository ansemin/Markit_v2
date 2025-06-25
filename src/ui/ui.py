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
from src.rag.vector_store import vector_store_manager
from src.services.data_clearing_service import data_clearing_service

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

def run_conversion_thread_multi(file_paths, parser_name, ocr_method_name, output_format, processing_type):
    """Run the conversion in a separate thread for multiple files."""
    import threading
    from src.services.document_service import DocumentService
    
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
    global conversion_cancelled
    
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
    
    # Format the content and wrap it in the scrollable container
    formatted_content = format_markdown_content(str(content))
    html_output = f"<div class='output-container'>{formatted_content}</div>"
    
    logger.info("Conversion completed successfully")
    
    # Auto-ingest the converted document for RAG
    try:
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

def handle_chat_message(message, history):
    """Handle a new chat message with streaming response."""
    if not message or not message.strip():
        return "", history, gr.update()
    
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
            # Update status in real-time during streaming
            updated_status = get_chat_status()
            yield "", history, updated_status
        
        logger.info(f"Chat response completed for message: {message[:50]}...")
        
        # Final status update after message completion
        final_status = get_chat_status()
        yield "", history, final_status
        
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
        # Update status even on error
        error_status = get_chat_status()
        yield "", history, error_status

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

def handle_clear_all_data():
    """Handle clearing all RAG data (vector store + chat history)."""
    try:
        # Clear all data using the data clearing service
        success, message, stats = data_clearing_service.clear_all_data()
        
        if success:
            # Reset chat session after clearing data
            session_id = rag_chat_service.start_new_session()
            
            # Get updated status
            updated_status = get_chat_status()
            
            # Create success message with stats
            if stats.get("total_cleared_documents", 0) > 0 or stats.get("total_cleared_files", 0) > 0:
                clear_msg = f"✅ {message}"
                session_msg = f"🆕 Started new session: {session_id}"
                combined_msg = f'{clear_msg}<br/><div class="session-info">{session_msg}</div>'
            else:
                combined_msg = f'ℹ️ {message}<br/><div class="session-info">🆕 Started new session: {session_id}</div>'
            
            logger.info(f"Data cleared successfully: {message}")
            
            return [], combined_msg, updated_status
        else:
            error_msg = f"❌ {message}"
            logger.error(f"Data clearing failed: {message}")
            
            # Still get updated status even on error
            updated_status = get_chat_status()
            
            return None, f'<div class="session-info">{error_msg}</div>', updated_status
            
    except Exception as e:
        error_msg = f"Error clearing data: {str(e)}"
        logger.error(error_msg)
        
        # Get current status
        current_status = get_chat_status()
        
        return None, f'<div class="session-info">❌ {error_msg}</div>', current_status

def handle_query_search(query, method, k_value):
    """Handle query search and return formatted results."""
    if not query or not query.strip():
        return """
        <div class="ranker-container">
            <div class="ranker-placeholder">
                <h3>🔍 Query Ranker</h3>
                <p>Enter a search query to find relevant document chunks with similarity scores.</p>
            </div>
        </div>
        """
    
    try:
        logger.info(f"Query search: '{query[:50]}...' using method: {method}")
        
        # Get results based on method
        results = []
        if method == "similarity":
            retriever = vector_store_manager.get_retriever("similarity", {"k": k_value})
            docs = retriever.invoke(query)
            # Try to get actual similarity scores
            try:
                vector_store = vector_store_manager.get_vector_store()
                if hasattr(vector_store, 'similarity_search_with_score'):
                    docs_with_scores = vector_store.similarity_search_with_score(query, k=k_value)
                    for i, (doc, score) in enumerate(docs_with_scores):
                        similarity_score = max(0, 1 - score) if score is not None else 0.8
                        results.append(_format_ranker_result(doc, similarity_score, i + 1))
                else:
                    # Fallback without scores
                    for i, doc in enumerate(docs):
                        score = 0.85 - (i * 0.05)
                        results.append(_format_ranker_result(doc, score, i + 1))
            except Exception as e:
                logger.warning(f"Could not get similarity scores: {e}")
                for i, doc in enumerate(docs):
                    score = 0.85 - (i * 0.05)
                    results.append(_format_ranker_result(doc, score, i + 1))
                    
        elif method == "mmr":
            retriever = vector_store_manager.get_retriever("mmr", {"k": k_value, "fetch_k": k_value * 2, "lambda_mult": 0.5})
            docs = retriever.invoke(query)
            for i, doc in enumerate(docs):
                results.append(_format_ranker_result(doc, None, i + 1))  # No score for MMR
                
        elif method == "bm25":
            retriever = vector_store_manager.get_bm25_retriever(k=k_value)
            docs = retriever.invoke(query)
            for i, doc in enumerate(docs):
                results.append(_format_ranker_result(doc, None, i + 1))  # No score for BM25
                
        elif method == "hybrid":
            retriever = vector_store_manager.get_hybrid_retriever(k=k_value, semantic_weight=0.7, keyword_weight=0.3)
            docs = retriever.invoke(query)
            # Explicitly limit results to k_value since EnsembleRetriever may return more
            docs = docs[:k_value]
            for i, doc in enumerate(docs):
                results.append(_format_ranker_result(doc, None, i + 1))  # No score for Hybrid
        
        return _format_ranker_results_html(results, query, method)
        
    except Exception as e:
        error_msg = f"Error during search: {str(e)}"
        logger.error(error_msg)
        return f"""
        <div class="ranker-container">
            <div class="ranker-error">
                <h3>❌ Search Error</h3>
                <p>{error_msg}</p>
                <p class="error-hint">Please check if documents are uploaded and the system is ready.</p>
            </div>
        </div>
        """

def _format_ranker_result(doc, score, rank):
    """Format a single document result for the ranker."""
    metadata = doc.metadata or {}
    
    # Extract metadata
    source = metadata.get("source", "Unknown Document")
    page = metadata.get("page", "N/A")
    chunk_id = metadata.get("chunk_id", f"chunk_{rank}")
    
    # Content length indicator
    content_length = len(doc.page_content)
    if content_length < 200:
        length_indicator = "📄 Short"
    elif content_length < 500:
        length_indicator = "📄 Medium"
    else:
        length_indicator = "📄 Long"
    
    # Rank-based confidence levels (applies to all methods)
    if rank <= 3:
        confidence = "High"
        confidence_color = "#22c55e"
        confidence_icon = "🟢"
    elif rank <= 6:
        confidence = "Medium"
        confidence_color = "#f59e0b"
        confidence_icon = "🟡"
    else:
        confidence = "Low"
        confidence_color = "#ef4444"
        confidence_icon = "🔴"
    
    result = {
        "rank": rank,
        "content": doc.page_content,
        "source": source,
        "page": page,
        "chunk_id": chunk_id,
        "length_indicator": length_indicator,
        "has_score": score is not None,
        "confidence": confidence,
        "confidence_color": confidence_color,
        "confidence_icon": confidence_icon
    }
    
    # Only add score if we have a real score (similarity search only)
    if score is not None:
        result["score"] = round(score, 3)
    
    return result

def _format_ranker_results_html(results, query, method):
    """Format search results as HTML."""
    if not results:
        return """
        <div class="ranker-container">
            <div class="ranker-no-results">
                <h3>🔍 No Results Found</h3>
                <p>No relevant documents found for your query.</p>
                <p class="no-results-hint">Try different keywords or check if documents are uploaded.</p>
            </div>
        </div>
        """
    
    # Method display names
    method_labels = {
        "similarity": "🎯 Similarity Search",
        "mmr": "🔀 MMR (Diverse)",
        "bm25": "🔍 BM25 (Keywords)",
        "hybrid": "🔗 Hybrid (Recommended)"
    }
    method_display = method_labels.get(method, method)
    
    # Start building HTML
    html_parts = [f"""
    <div class="ranker-container">
        <div class="ranker-header">
            <div class="ranker-title">
                <h3>🔍 Search Results</h3>
                <div class="query-display">"{query}"</div>
            </div>
            <div class="ranker-meta">
                <span class="method-badge">{method_display}</span>
                <span class="result-count">{len(results)} results</span>
            </div>
        </div>
    """]
    
    # Add results
    for result in results:
        rank_emoji = ["🥇", "🥈", "🥉"][result["rank"] - 1] if result["rank"] <= 3 else f"#{result['rank']}"
        
        # Escape content for safe HTML inclusion and JavaScript
        escaped_content = result['content'].replace('"', '&quot;').replace("'", "&#39;").replace('\n', '\\n')
        
        # Build score info - always show confidence, only show score for similarity search
        score_info_parts = [f"""
                    <span class="confidence-badge" style="color: {result['confidence_color']}">
                        {result['confidence_icon']} {result['confidence']}
                    </span>"""]
        
        # Only add score value if we have real scores (similarity search)
        if result.get('has_score', False):
            score_info_parts.append(f'<span class="score-value">🎯 {result["score"]}</span>')
        
        score_info_html = f"""
                <div class="score-info">
                    {''.join(score_info_parts)}
                </div>"""
        
        html_parts.append(f"""
        <div class="result-card">
            <div class="result-header">
                <div class="rank-info">
                    <span class="rank-badge">{rank_emoji} Rank {result['rank']}</span>
                    <span class="source-info">📄 {result['source']}</span>
                    {f"<span class='page-info'>Page {result['page']}</span>" if result['page'] != 'N/A' else ""}
                    <span class="length-info">{result['length_indicator']}</span>
                </div>
                {score_info_html}
            </div>
            <div class="result-content">
                <div class="content-text">{result['content']}</div>
            </div>
        </div>
        """)
    
    html_parts.append("</div>")
    
    return "".join(html_parts)

def get_ranker_status():
    """Get current ranker system status."""
    try:
        # Get collection info
        collection_info = vector_store_manager.get_collection_info()
        document_count = collection_info.get("document_count", 0)
        
        # Get available methods
        available_methods = ["similarity", "mmr", "bm25", "hybrid"]
        
        # Check if system is ready
        ingestion_status = document_ingestion_service.get_ingestion_status()
        system_ready = ingestion_status.get('system_ready', False)
        
        status_html = f"""
        <div class="status-card">
            <div class="status-header">
                <h3>🔍 Query Ranker Status</h3>
                <div class="status-indicator {'status-ready' if system_ready else 'status-not-ready'}">
                    {'🟢 READY' if system_ready else '🔴 NOT READY'}
                </div>
            </div>
            
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-label">Available Documents</div>
                    <div class="status-value">{document_count}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Retrieval Methods</div>
                    <div class="status-value">{len(available_methods)}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Vector Store</div>
                    <div class="status-value">{'Ready' if system_ready else 'Not Ready'}</div>
                </div>
            </div>
            
            <div class="ranker-methods">
                <div class="methods-label">Available Methods:</div>
                <div class="methods-list">
                    <span class="method-tag">🎯 Similarity</span>
                    <span class="method-tag">🔀 MMR</span>
                    <span class="method-tag">🔍 BM25</span>
                    <span class="method-tag">🔗 Hybrid</span>
                </div>
            </div>
        </div>
        """
        
        return status_html
        
    except Exception as e:
        error_msg = f"Error getting ranker status: {str(e)}"
        logger.error(error_msg)
        return f"""
        <div class="status-card status-error">
            <div class="status-header">
                <h3>❌ System Error</h3>
            </div>
            <p class="error-message">{error_msg}</p>
        </div>
        """

def get_chat_status():
    """Get current chat system status."""
    try:
        # Check ingestion status
        ingestion_status = document_ingestion_service.get_ingestion_status()
        
        # Check usage stats
        usage_stats = rag_chat_service.get_usage_stats()
        
        # Get data status for additional context
        data_status = data_clearing_service.get_data_status()
        
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
                    <div class="status-label">Vector Store Docs</div>
                    <div class="status-value">{data_status.get('vector_store', {}).get('document_count', 0)}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Chat History Files</div>
                    <div class="status-value">{data_status.get('chat_history', {}).get('file_count', 0)}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Session Usage</div>
                    <div class="status-value">{usage_stats.get('session_messages', 0)}/{usage_stats.get('session_limit', 50)}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Environment</div>
                    <div class="status-value">{'HF Space' if data_status.get('environment') == 'hf_space' else 'Local'}</div>
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
            color: #2c3e50 !important;
        }
        
        .service-status span {
            color: #2c3e50 !important;
        }
        
        .service-ready {
            background: #d4edda;
            color: #2c3e50 !important;
            border: 1px solid #c3e6cb;
        }
        
        .service-ready span {
            color: #2c3e50 !important;
        }
        
        .service-error {
            background: #f8d7da;
            color: #2c3e50 !important;
            border: 1px solid #f5c6cb;
        }
        
        .service-error span {
            color: #2c3e50 !important;
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
        
        .btn-clear-data {
            background: #dc3545;
            color: white;
        }
        
        .btn-clear-data:hover {
            background: #c82333;
            transform: translateY(-1px);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
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
        
        /* Query Ranker Styles */
        .ranker-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .ranker-placeholder {
            text-align: center;
            padding: 40px;
            background: #f8f9fa;
            border-radius: 12px;
            border: 1px solid #e9ecef;
            color: #6c757d;
        }
        
        .ranker-placeholder h3 {
            color: #495057;
            margin-bottom: 10px;
        }
        
        .ranker-error {
            text-align: center;
            padding: 30px;
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 12px;
            color: #721c24;
        }
        
        .ranker-error h3 {
            margin-bottom: 15px;
        }
        
        .error-hint {
            font-style: italic;
            margin-top: 10px;
            opacity: 0.8;
        }
        
        .ranker-no-results {
            text-align: center;
            padding: 40px;
            background: #ffffff;
            border: 1px solid #e1e5e9;
            border-radius: 12px;
            color: #6c757d;
        }
        
        .ranker-no-results h3 {
            color: #495057;
            margin-bottom: 15px;
        }
        
        .no-results-hint {
            font-style: italic;
            margin-top: 10px;
            opacity: 0.8;
        }
        
        .ranker-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .ranker-title h3 {
            margin: 0 0 10px 0;
            font-size: 1.4em;
            font-weight: 600;
        }
        
        .query-display {
            font-size: 1.1em;
            opacity: 0.9;
            font-style: italic;
            margin-bottom: 15px;
        }
        
        .ranker-meta {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .method-badge {
            background: rgba(255, 255, 255, 0.2);
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 500;
            font-size: 0.9em;
        }
        
        .result-count {
            background: rgba(255, 255, 255, 0.15);
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 500;
            font-size: 0.9em;
        }
        
        .result-card {
            background: #ffffff;
            border: 1px solid #e1e5e9;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            overflow: hidden;
        }
        
        .result-card:hover {
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }
        
        .rank-info {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .rank-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 10px;
            border-radius: 15px;
            font-weight: 600;
            font-size: 0.85em;
        }
        
        .source-info {
            background: #e9ecef;
            color: #495057;
            padding: 4px 8px;
            border-radius: 10px;
            font-size: 0.85em;
            font-weight: 500;
        }
        
        .page-info {
            background: #d1ecf1;
            color: #0c5460;
            padding: 4px 8px;
            border-radius: 10px;
            font-size: 0.85em;
        }
        
        .length-info {
            background: #f8f9fa;
            color: #6c757d;
            padding: 4px 8px;
            border-radius: 10px;
            font-size: 0.85em;
        }
        
        .score-info {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .confidence-badge {
            padding: 4px 8px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.85em;
        }
        
        .score-value {
            background: #2c3e50;
            color: white;
            padding: 6px 12px;
            border-radius: 15px;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .result-content {
            padding: 20px;
        }
        
        .content-text {
            line-height: 1.6;
            color: #2c3e50;
            border-left: 3px solid #667eea;
            padding-left: 15px;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .result-actions {
            display: flex;
            gap: 10px;
            padding: 15px 20px;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
        }
        
        .action-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9em;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .copy-btn {
            background: #17a2b8;
            color: white;
        }
        
        .copy-btn:hover {
            background: #138496;
            transform: translateY(-1px);
        }
        
        .info-btn {
            background: #6c757d;
            color: white;
        }
        
        .info-btn:hover {
            background: #5a6268;
            transform: translateY(-1px);
        }
        
        .ranker-methods {
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #e9ecef;
        }
        
        .methods-label {
            font-weight: 600;
            color: #495057;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        
        .methods-list {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        .method-tag {
            background: #e9ecef;
            color: #495057;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
        }
        
        /* Ranker controls styling */
        .ranker-controls {
            background: #ffffff;
            border: 1px solid #e1e5e9;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .ranker-input-row {
            display: flex;
            gap: 15px;
            align-items: end;
            margin-bottom: 15px;
        }
        
        .ranker-query-input {
            flex: 1;
            border: 2px solid #e1e5e9;
            border-radius: 25px;
            padding: 12px 20px;
            font-size: 1em;
            transition: all 0.3s ease;
        }
        
        .ranker-query-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            outline: none;
        }
        
        .ranker-search-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            min-width: 100px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            font-size: 1em;
        }
        
        .ranker-search-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .ranker-options-row {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        /* Responsive design for ranker */
        @media (max-width: 768px) {
            .ranker-container {
                padding: 10px;
            }
            
            .ranker-input-row {
                flex-direction: column;
                gap: 10px;
            }
            
            .ranker-options-row {
                flex-direction: column;
                gap: 10px;
                align-items: stretch;
            }
            
            .ranker-meta {
                justify-content: center;
            }
            
            .rank-info {
                flex-direction: column;
                gap: 5px;
                align-items: flex-start;
            }
            
            .result-header {
                flex-direction: column;
                gap: 10px;
                align-items: flex-start;
            }
            
            .score-info {
                align-self: flex-end;
            }
            
            .result-actions {
                flex-direction: column;
                gap: 8px;
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
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".txt", ".md", ".html", ".htm"]
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

                # Event handlers for document converter
                
                # Update UI when files are uploaded/changed
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
                        clear_data_btn = gr.Button("🗑️ Clear All Data", elem_classes=["control-btn", "btn-clear-data"], variant="stop")
                    
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
                    outputs=[msg_input, chatbot, status_display]
                )
                
                send_btn.click(
                    fn=handle_chat_message,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot, status_display]
                )
                
                # New session handler with improved feedback
                def enhanced_new_session():
                    history, info = start_new_chat_session()
                    session_html = f'<div class="session-info">{info}</div>'
                    updated_status = get_chat_status()
                    return history, session_html, updated_status
                
                new_session_btn.click(
                    fn=enhanced_new_session,
                    inputs=[],
                    outputs=[chatbot, session_info, status_display]
                )
                
                # Refresh status handler
                refresh_status_btn.click(
                    fn=get_chat_status,
                    inputs=[],
                    outputs=[status_display]
                )
                
                # Clear all data handler
                clear_data_btn.click(
                    fn=handle_clear_all_data,
                    inputs=[],
                    outputs=[chatbot, session_info, status_display]
                )

            # Query Ranker Tab
            with gr.TabItem("🔍 Query Ranker"):
                with gr.Column(elem_classes=["ranker-container"]):
                    # Modern header
                    gr.HTML("""
                    <div class="chat-header">
                        <h2>🔍 Query Ranker</h2>
                        <p>Search and rank document chunks with similarity scores</p>
                    </div>
                    """)
                    
                    # Status section
                    ranker_status_display = gr.HTML(value=get_ranker_status())
                    
                    # Control buttons
                    with gr.Row(elem_classes=["control-buttons"]):
                        refresh_ranker_status_btn = gr.Button("🔄 Refresh Status", elem_classes=["control-btn", "btn-refresh"])
                        clear_results_btn = gr.Button("🗑️ Clear Results", elem_classes=["control-btn", "btn-clear-data"])
                    
                    # Search controls
                    with gr.Column(elem_classes=["ranker-controls"]):
                        with gr.Row(elem_classes=["ranker-input-row"]):
                            query_input = gr.Textbox(
                                placeholder="Enter your search query...",
                                show_label=False,
                                elem_classes=["ranker-query-input"],
                                scale=4
                            )
                            search_btn = gr.Button("🔍 Search", elem_classes=["ranker-search-btn"], scale=0)
                        
                        with gr.Row(elem_classes=["ranker-options-row"]):
                            method_dropdown = gr.Dropdown(
                                choices=[
                                    ("🎯 Similarity Search", "similarity"),
                                    ("🔀 MMR (Diverse)", "mmr"),
                                    ("🔍 BM25 (Keywords)", "bm25"),
                                    ("🔗 Hybrid (Recommended)", "hybrid")
                                ],
                                value="hybrid",
                                label="Retrieval Method",
                                scale=2
                            )
                            k_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Number of Results",
                                scale=1
                            )
                    
                    # Results display
                    results_display = gr.HTML(
                        value=handle_query_search("", "hybrid", 5),  # Initial placeholder
                        elem_classes=["ranker-results-container"]
                    )
                
                # Event handlers for Query Ranker
                def clear_ranker_results():
                    """Clear the search results and reset to placeholder."""
                    return handle_query_search("", "hybrid", 5), ""
                
                def refresh_ranker_status():
                    """Refresh the ranker status display."""
                    return get_ranker_status()
                
                # Search functionality
                query_input.submit(
                    fn=handle_query_search,
                    inputs=[query_input, method_dropdown, k_slider],
                    outputs=[results_display]
                )
                
                search_btn.click(
                    fn=handle_query_search,
                    inputs=[query_input, method_dropdown, k_slider],
                    outputs=[results_display]
                )
                
                # Control button handlers
                refresh_ranker_status_btn.click(
                    fn=refresh_ranker_status,
                    inputs=[],
                    outputs=[ranker_status_display]
                )
                
                clear_results_btn.click(
                    fn=clear_ranker_results,
                    inputs=[],
                    outputs=[results_display, query_input]
                )
                
                # Update results when method or k changes
                method_dropdown.change(
                    fn=handle_query_search,
                    inputs=[query_input, method_dropdown, k_slider],
                    outputs=[results_display]
                )
                
                k_slider.change(
                    fn=handle_query_search,
                    inputs=[query_input, method_dropdown, k_slider],
                    outputs=[results_display]
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