"""Chat history and memory management for RAG conversations."""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.core.config import config
from src.core.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class ChatMessage:
    """Represents a single chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    sources: Optional[List[str]] = None  # Source documents used for the response
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class ChatSession:
    """Represents a chat session with history."""
    session_id: str
    created_at: str
    updated_at: str
    messages: List[ChatMessage]
    document_sources: List[str]  # Documents available in this session
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [msg.to_dict() for msg in self.messages],
            "document_sources": self.document_sources
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create from dictionary."""
        messages = [ChatMessage.from_dict(msg) for msg in data.get("messages", [])]
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            messages=messages,
            document_sources=data.get("document_sources", [])
        )

class ChatMemoryManager:
    """Manages chat history and memory for RAG conversations."""
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the chat memory manager.
        
        Args:
            persist_directory: Directory to persist chat history
        """
        if persist_directory is None:
            persist_directory = config.rag.chat_history_path
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.current_session: Optional[ChatSession] = None
        
        logger.info(f"ChatMemoryManager initialized with persist_directory={self.persist_directory}")
    
    def create_session(self, document_sources: Optional[List[str]] = None) -> str:
        """
        Create a new chat session.
        
        Args:
            document_sources: List of document sources available for this session
            
        Returns:
            Session ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        now = datetime.now().isoformat()
        
        self.current_session = ChatSession(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            messages=[],
            document_sources=document_sources or []
        )
        
        logger.info(f"Created new chat session: {session_id}")
        return session_id
    
    def add_message(self, role: str, content: str, sources: Optional[List[str]] = None) -> None:
        """
        Add a message to the current session.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            sources: Source documents used (for assistant messages)
        """
        if self.current_session is None:
            self.create_session()
        
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            sources=sources
        )
        
        self.current_session.messages.append(message)
        self.current_session.updated_at = datetime.now().isoformat()
        
        logger.info(f"Added {role} message to session {self.current_session.session_id}")
    
    def get_conversation_history(self, max_messages: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Get conversation history in Gradio chat format.
        
        Args:
            max_messages: Maximum number of messages to return
            
        Returns:
            List of (user_message, assistant_message) tuples
        """
        if not self.current_session or not self.current_session.messages:
            return []
        
        messages = self.current_session.messages
        if max_messages:
            messages = messages[-max_messages:]
        
        # Group messages into pairs
        history = []
        user_msg = None
        
        for msg in messages:
            if msg.role == "user":
                user_msg = msg.content
            elif msg.role == "assistant" and user_msg is not None:
                history.append((user_msg, msg.content))
                user_msg = None
        
        return history
    
    def get_context_messages(self, max_context_length: int = 4000) -> List[BaseMessage]:
        """
        Get recent messages formatted for LangChain context.
        
        Args:
            max_context_length: Maximum context length in characters
            
        Returns:
            List of LangChain message objects
        """
        if not self.current_session or not self.current_session.messages:
            return []
        
        context_messages = []
        current_length = 0
        
        # Start from the most recent messages and work backwards
        for msg in reversed(self.current_session.messages):
            msg_length = len(msg.content)
            
            if current_length + msg_length > max_context_length:
                break
            
            if msg.role == "user":
                context_messages.insert(0, HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                context_messages.insert(0, AIMessage(content=msg.content))
            
            current_length += msg_length
        
        logger.info(f"Retrieved {len(context_messages)} context messages ({current_length} chars)")
        return context_messages
    
    def save_session(self) -> bool:
        """
        Save the current session to disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.current_session:
            return False
        
        try:
            session_file = self.persist_directory / f"{self.current_session.session_id}.json"
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_session.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved session {self.current_session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False
    
    def load_session(self, session_id: str) -> bool:
        """
        Load a session from disk.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_file = self.persist_directory / f"{session_id}.json"
            
            if not session_file.exists():
                logger.warning(f"Session file not found: {session_id}")
                return False
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.current_session = ChatSession.from_dict(session_data)
            logger.info(f"Loaded session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all saved sessions.
        
        Returns:
            List of session metadata
        """
        sessions = []
        
        try:
            for session_file in self.persist_directory.glob("session_*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    sessions.append({
                        "session_id": session_data["session_id"],
                        "created_at": session_data["created_at"],
                        "updated_at": session_data["updated_at"],
                        "message_count": len(session_data.get("messages", [])),
                        "document_sources": session_data.get("document_sources", [])
                    })
                    
                except Exception as e:
                    logger.warning(f"Error reading session file {session_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
        
        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        return sessions
    
    def clear_current_session(self) -> None:
        """Clear the current session."""
        self.current_session = None
        logger.info("Cleared current session")

# Global chat memory manager instance
chat_memory_manager = ChatMemoryManager()