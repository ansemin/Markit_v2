"""Chat interface UI component and logic."""

import gradio as gr
import logging

from src.core.logging_config import get_logger
from src.rag import rag_chat_service
from src.services.data_clearing_service import data_clearing_service

logger = get_logger(__name__)


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


def get_chat_status():
    """Get current chat system status."""
    try:
        # Check ingestion status
        from src.rag import document_ingestion_service
        from src.services.data_clearing_service import data_clearing_service
        
        ingestion_status = document_ingestion_service.get_ingestion_status()
        
        # Check usage stats
        usage_stats = rag_chat_service.get_usage_stats()
        
        # Get data status for additional context
        data_status = data_clearing_service.get_data_status()
        
        # Get environment info
        import os
        env_type = "Hugging Face Space" if os.getenv("SPACE_ID") else "Local Development"
        
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


def create_chat_interface_tab():
    """Create the chat interface tab UI."""
    with gr.TabItem("💬 Chat with Documents"):
        with gr.Column(elem_classes=["chat-tab-container"]):
            # Header
            gr.HTML("""
            <div class="chat-header">
                <h2>💬 Chat with your converted documents</h2>
                <p>Ask questions about your documents using advanced RAG technology</p>
            </div>
            """)
            
            # Status monitoring
            status_display = gr.HTML(value=get_chat_status())
            
            # Control buttons
            with gr.Row(elem_classes=["control-buttons"]):
                refresh_btn = gr.Button("🔄 Refresh Status", elem_classes=["control-btn", "btn-refresh"])
                new_session_btn = gr.Button("🆕 New Session", elem_classes=["control-btn", "btn-new-session"])
                clear_data_btn = gr.Button("🗑️ Clear All Data", elem_classes=["control-btn", "btn-clear-data"], variant="stop")
            
            # Chat interface
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
            
            # Session info display
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
            
            # Control button handlers
            refresh_btn.click(
                fn=get_chat_status,
                inputs=[],
                outputs=[status_display]
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
            
            clear_data_btn.click(
                handle_clear_all_data,
                outputs=[chatbot, session_info, status_display]
            )