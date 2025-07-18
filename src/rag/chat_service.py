"""RAG chat service with Gemini 2.5 Flash and streaming support."""

import os
import time
from typing import List, Dict, Any, Optional, Generator, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.rag.vector_store import vector_store_manager
from src.rag.memory import chat_memory_manager
from src.core.config import config
from src.core.logging_config import get_logger

logger = get_logger(__name__)

class ChatUsageLimiter:
    """Manages chat usage limits to prevent abuse."""
    
    def __init__(self, max_messages_per_session: int = 50, max_messages_per_hour: int = 100):
        """
        Initialize usage limiter.
        
        Args:
            max_messages_per_session: Maximum messages per chat session
            max_messages_per_hour: Maximum messages per hour across all sessions
        """
        self.max_messages_per_session = max_messages_per_session
        self.max_messages_per_hour = max_messages_per_hour
        self.hourly_usage = {}  # Track usage by hour
        
        logger.info(f"Chat usage limiter initialized: {max_messages_per_session}/session, {max_messages_per_hour}/hour")
    
    def check_session_limit(self, session_message_count: int) -> Tuple[bool, str]:
        """
        Check if session has exceeded message limit.
        
        Args:
            session_message_count: Number of messages in current session
            
        Returns:
            Tuple of (allowed, reason_if_not_allowed)
        """
        if session_message_count >= self.max_messages_per_session:
            return False, f"Session limit reached ({self.max_messages_per_session} messages per session). Please start a new chat."
        return True, ""
    
    def check_hourly_limit(self) -> Tuple[bool, str]:
        """
        Check if hourly limit has been exceeded.
        
        Returns:
            Tuple of (allowed, reason_if_not_allowed)
        """
        current_hour = int(time.time()) // 3600
        
        # Clean old entries (keep only last 2 hours)
        hours_to_keep = [current_hour - 1, current_hour]
        self.hourly_usage = {h: count for h, count in self.hourly_usage.items() if h in hours_to_keep}
        
        current_usage = self.hourly_usage.get(current_hour, 0)
        
        if current_usage >= self.max_messages_per_hour:
            return False, f"Hourly limit reached ({self.max_messages_per_hour} messages per hour). Please try again later."
        
        return True, ""
    
    def record_usage(self) -> None:
        """Record a message usage."""
        current_hour = int(time.time()) // 3600
        self.hourly_usage[current_hour] = self.hourly_usage.get(current_hour, 0) + 1
    
    def can_send_message(self, session_message_count: int) -> Tuple[bool, str]:
        """
        Check if user can send a message.
        
        Args:
            session_message_count: Number of messages in current session
            
        Returns:
            Tuple of (allowed, reason_if_not_allowed)
        """
        # Check session limit
        session_ok, session_reason = self.check_session_limit(session_message_count)
        if not session_ok:
            return False, session_reason
        
        # Check hourly limit
        hourly_ok, hourly_reason = self.check_hourly_limit()
        if not hourly_ok:
            return False, hourly_reason
        
        return True, ""

class RAGChatService:
    """RAG-powered chat service with document context."""
    
    def __init__(self):
        """Initialize the RAG chat service."""
        self.usage_limiter = ChatUsageLimiter(
            max_messages_per_session=config.rag.max_messages_per_session,
            max_messages_per_hour=config.rag.max_messages_per_hour
        )
        self._llm = None
        self._rag_chain = None
        self._current_retrieval_method = "similarity"
        self._default_retrieval_method = "similarity"
        self._default_retrieval_config = {"k": 4}
        
        logger.info("RAG chat service initialized")
    
    def get_llm(self) -> ChatGoogleGenerativeAI:
        """Get or create the Gemini LLM instance."""
        if self._llm is None:
            try:
                google_api_key = config.api.google_api_key or os.getenv("GOOGLE_API_KEY")
                
                if not google_api_key:
                    raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in environment variables.")
                
                self._llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",  # Latest Gemini model
                    google_api_key=google_api_key,
                    temperature=config.rag.rag_temperature,
                    max_tokens=config.rag.rag_max_tokens,
                    disable_streaming=False  # Enable streaming (new parameter name)
                )
                
                logger.info("Gemini 2.5 Flash LLM initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Gemini LLM: {e}")
                raise
        
        return self._llm
    
    def create_rag_chain(self, retrieval_method: str = "similarity", retrieval_config: Optional[Dict[str, Any]] = None):
        """
        Create the RAG chain for document-aware conversations.
        
        Args:
            retrieval_method: Method to use ("similarity", "mmr", "hybrid")
            retrieval_config: Configuration for the retrieval method
        """
        if self._rag_chain is None or hasattr(self, '_current_retrieval_method') and self._current_retrieval_method != retrieval_method:
            try:
                llm = self.get_llm()
                
                # Set default retrieval config
                if retrieval_config is None:
                    retrieval_config = {"k": 4}
                
                # Get retriever based on method
                if retrieval_method == "hybrid":
                    # Use hybrid retriever (semantic + keyword)
                    semantic_weight = retrieval_config.get("semantic_weight", 0.7)
                    keyword_weight = retrieval_config.get("keyword_weight", 0.3)
                    search_type = retrieval_config.get("search_type", "similarity")
                    search_kwargs = {k: v for k, v in retrieval_config.items() 
                                   if k not in ["semantic_weight", "keyword_weight", "search_type"]}
                    
                    retriever = vector_store_manager.get_hybrid_retriever(
                        k=retrieval_config.get("k", 4),
                        semantic_weight=semantic_weight,
                        keyword_weight=keyword_weight,
                        search_type=search_type,
                        search_kwargs=search_kwargs if search_kwargs else None
                    )
                    logger.info(f"Using hybrid retriever with weights: semantic={semantic_weight}, keyword={keyword_weight}")
                    
                elif retrieval_method == "mmr":
                    # Use MMR for diversity
                    search_kwargs = retrieval_config.copy()
                    if "fetch_k" not in search_kwargs:
                        search_kwargs["fetch_k"] = retrieval_config.get("k", 4) * 5  # Default fetch 5x more for MMR
                    if "lambda_mult" not in search_kwargs:
                        search_kwargs["lambda_mult"] = 0.5  # Balance relevance vs diversity
                    
                    retriever = vector_store_manager.get_retriever(
                        search_type="mmr",
                        search_kwargs=search_kwargs
                    )
                    logger.info(f"Using MMR retriever with config: {search_kwargs}")
                    
                else:
                    # Default similarity search
                    retriever = vector_store_manager.get_retriever(
                        search_type="similarity",
                        search_kwargs=retrieval_config
                    )
                    logger.info(f"Using similarity retriever with config: {retrieval_config}")
                
                # Store current method for comparison
                self._current_retrieval_method = retrieval_method
                
                # Create a prompt template for RAG
                prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant that can chat naturally while specializing in answering questions about uploaded documents.

Instructions:
1. For document-related questions: Use the provided context to give comprehensive answers and always cite your sources
2. For conversational interactions (greetings, introductions, clarifications, follow-ups): Respond naturally and helpfully
3. For questions about topics not covered in the documents: Politely explain that you specialize in the uploaded documents but can still have a conversation
4. When using document information, always cite which parts of the documents you referenced
5. Include relevant tables and code blocks when they help answer the question
6. Be conversational, friendly, and helpful
7. Remember information shared in our conversation (like names, preferences, etc.)

Context from documents:
{context}

Chat History:
{chat_history}

User Message: {question}
""")
                
                def format_docs(docs: List[Document]) -> str:
                    """Format retrieved documents for context."""
                    if not docs:
                        return "No relevant documents found."
                    
                    formatted = []
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get('source', 'Unknown')
                        chunk_id = doc.metadata.get('chunk_id', f'chunk_{i}')
                        
                        formatted.append(f"Document {i} (Source: {source}, ID: {chunk_id}):\n{doc.page_content}")
                    
                    return "\n\n".join(formatted)
                
                def format_chat_history() -> str:
                    """Format chat history for context."""
                    history = chat_memory_manager.get_conversation_history(max_messages=10)
                    if not history:
                        return "No previous conversation."
                    
                    formatted = []
                    for user_msg, assistant_msg in history[-5:]:  # Last 5 exchanges
                        formatted.append(f"User: {user_msg}")
                        formatted.append(f"Assistant: {assistant_msg}")
                    
                    return "\n".join(formatted)
                
                # Create the RAG chain
                self._rag_chain = (
                    {
                        "context": retriever | format_docs,
                        "chat_history": lambda _: format_chat_history(),
                        "question": RunnablePassthrough()
                    }
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                logger.info("RAG chain created successfully")
                
            except Exception as e:
                logger.error(f"Failed to create RAG chain: {e}")
                raise
    
    def get_rag_chain(self, retrieval_method: str = "similarity", retrieval_config: Optional[Dict[str, Any]] = None):
        """
        Get the RAG chain, creating it if necessary.
        
        Args:
            retrieval_method: Method to use ("similarity", "mmr", "hybrid")
            retrieval_config: Configuration for the retrieval method
        """
        if self._rag_chain is None or (hasattr(self, '_current_retrieval_method') and self._current_retrieval_method != retrieval_method):
            self.create_rag_chain(retrieval_method, retrieval_config)
        return self._rag_chain
    
    def chat_stream_with_retrieval(self, user_message: str, retrieval_method: str = "similarity", retrieval_config: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """
        Stream chat response using RAG with specified retrieval method.
        
        Args:
            user_message: User's message
            retrieval_method: Method to use ("similarity", "mmr", "hybrid")
            retrieval_config: Configuration for the retrieval method
            
        Yields:
            Chunks of the response as they're generated
        """
        try:
            # Check usage limits
            current_session = chat_memory_manager.current_session
            session_message_count = len(current_session.messages) if current_session else 0
            
            can_send, reason = self.usage_limiter.can_send_message(session_message_count)
            if not can_send:
                yield f"❌ {reason}"
                return
            
            # Record usage
            self.usage_limiter.record_usage()
            
            # Add user message to memory
            chat_memory_manager.add_message("user", user_message)
            
            # Get RAG chain with specified retrieval method
            rag_chain = self.get_rag_chain(retrieval_method, retrieval_config)
            
            # Stream the response
            response_chunks = []
            for chunk in rag_chain.stream(user_message):
                if chunk:
                    response_chunks.append(chunk)
                    yield chunk
            
            # Save complete response to memory
            complete_response = "".join(response_chunks)
            if complete_response.strip():
                chat_memory_manager.add_message("assistant", complete_response)
                
                # Save session periodically
                chat_memory_manager.save_session()
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            yield f"❌ {error_msg}"
    
    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """
        Stream chat response using RAG.
        
        Args:
            user_message: User's message
            
        Yields:
            Chunks of the response as they're generated
        """
        try:
            # Check usage limits
            current_session = chat_memory_manager.current_session
            session_message_count = len(current_session.messages) if current_session else 0
            
            can_send, reason = self.usage_limiter.can_send_message(session_message_count)
            if not can_send:
                yield f"❌ {reason}"
                return
            
            # Record usage
            self.usage_limiter.record_usage()
            
            # Add user message to memory
            chat_memory_manager.add_message("user", user_message)
            
            # Get RAG chain
            rag_chain = self.get_rag_chain()
            
            # Stream the response
            response_chunks = []
            for chunk in rag_chain.stream(user_message):
                if chunk:
                    response_chunks.append(chunk)
                    yield chunk
            
            # Save complete response to memory
            complete_response = "".join(response_chunks)
            if complete_response.strip():
                chat_memory_manager.add_message("assistant", complete_response)
                
                # Save session periodically
                chat_memory_manager.save_session()
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            yield f"❌ {error_msg}"
    
    def chat(self, user_message: str) -> str:
        """
        Get a complete chat response (non-streaming).
        
        Args:
            user_message: User's message
            
        Returns:
            Complete response string
        """
        try:
            # Check usage limits
            current_session = chat_memory_manager.current_session
            session_message_count = len(current_session.messages) if current_session else 0
            
            can_send, reason = self.usage_limiter.can_send_message(session_message_count)
            if not can_send:
                return f"❌ {reason}"
            
            # Record usage
            self.usage_limiter.record_usage()
            
            # Add user message to memory
            chat_memory_manager.add_message("user", user_message)
            
            # Get RAG chain
            rag_chain = self.get_rag_chain()
            
            # Get response
            response = rag_chain.invoke(user_message)
            
            # Save response to memory
            if response.strip():
                chat_memory_manager.add_message("assistant", response)
                chat_memory_manager.save_session()
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def chat_with_retrieval(self, user_message: str, retrieval_method: str = "similarity", retrieval_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Get a complete chat response with specified retrieval method (non-streaming).
        
        Args:
            user_message: User's message
            retrieval_method: Method to use ("similarity", "mmr", "hybrid")
            retrieval_config: Configuration for the retrieval method
            
        Returns:
            Complete response string
        """
        try:
            # Check usage limits
            current_session = chat_memory_manager.current_session
            session_message_count = len(current_session.messages) if current_session else 0
            
            can_send, reason = self.usage_limiter.can_send_message(session_message_count)
            if not can_send:
                return f"❌ {reason}"
            
            # Record usage
            self.usage_limiter.record_usage()
            
            # Add user message to memory
            chat_memory_manager.add_message("user", user_message)
            
            # Get RAG chain with specified retrieval method
            rag_chain = self.get_rag_chain(retrieval_method, retrieval_config)
            
            # Get response
            response = rag_chain.invoke(user_message)
            
            # Save response to memory
            if response.strip():
                chat_memory_manager.add_message("assistant", response)
                chat_memory_manager.save_session()
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def set_default_retrieval_method(self, method: str, config: Optional[Dict[str, Any]] = None):
        """
        Set the default retrieval method for this service.
        
        Args:
            method: Retrieval method ("similarity", "mmr", "hybrid")
            config: Configuration for the method
        """
        self._default_retrieval_method = method
        self._default_retrieval_config = config or {}
        
        # Reset the chain to use new method
        self._rag_chain = None
        
        logger.info(f"Default retrieval method set to: {method} with config: {config}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        current_session = chat_memory_manager.current_session
        session_message_count = len(current_session.messages) if current_session else 0
        
        current_hour = int(time.time()) // 3600
        hourly_count = self.usage_limiter.hourly_usage.get(current_hour, 0)
        
        return {
            "session_messages": session_message_count,
            "session_limit": self.usage_limiter.max_messages_per_session,
            "hourly_messages": hourly_count,
            "hourly_limit": self.usage_limiter.max_messages_per_hour,
            "session_remaining": max(0, self.usage_limiter.max_messages_per_session - session_message_count),
            "hourly_remaining": max(0, self.usage_limiter.max_messages_per_hour - hourly_count)
        }
    
    def start_new_session(self, document_sources: Optional[List[str]] = None) -> str:
        """Start a new chat session."""
        session_id = chat_memory_manager.create_session(document_sources)
        logger.info(f"Started new chat session: {session_id}")
        return session_id
    
    def test_service(self) -> Dict[str, Any]:
        """Test the RAG service components."""
        results = {
            "llm_available": False,
            "vector_store_available": False,
            "embeddings_available": False,
            "errors": []
        }
        
        try:
            # Test LLM
            llm = self.get_llm()
            test_response = llm.invoke("Test message")
            results["llm_available"] = True
        except Exception as e:
            results["errors"].append(f"LLM test failed: {str(e)}")
        
        try:
            # Test vector store
            vector_info = vector_store_manager.get_collection_info()
            results["vector_store_available"] = "error" not in vector_info
            results["document_count"] = vector_info.get("document_count", 0)
        except Exception as e:
            results["errors"].append(f"Vector store test failed: {str(e)}")
        
        try:
            # Test embeddings
            from src.rag.embeddings import embedding_manager
            results["embeddings_available"] = embedding_manager.test_embedding_model()
        except Exception as e:
            results["errors"].append(f"Embeddings test failed: {str(e)}")
        
        return results

# Global RAG chat service instance
rag_chat_service = RAGChatService()