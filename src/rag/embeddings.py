"""Embedding model management for RAG functionality."""

import os
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.core.config import config
from src.core.logging_config import get_logger

logger = get_logger(__name__)

class EmbeddingManager:
    """Manages embedding models for document vectorization."""
    
    def __init__(self):
        self._embedding_model: Optional[GoogleGenerativeAIEmbeddings] = None
        
    def get_embedding_model(self) -> GoogleGenerativeAIEmbeddings:
        """Get or create the Gemini embedding model."""
        if self._embedding_model is None:
            try:
                # Get Google API key from config/environment
                google_api_key = config.api.google_api_key or os.getenv("GOOGLE_API_KEY")
                
                if not google_api_key:
                    raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in environment variables.")
                
                self._embedding_model = GoogleGenerativeAIEmbeddings(
                    model=config.rag.embedding_model,
                    google_api_key=google_api_key,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                
                logger.info(f"Gemini embedding model ({config.rag.embedding_model}) initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Gemini embedding model: {e}")
                raise
                
        return self._embedding_model
    
    def test_embedding_model(self) -> bool:
        """Test if the embedding model is working correctly."""
        try:
            embedding_model = self.get_embedding_model()
            # Test with a simple text
            test_text = "This is a test for embedding functionality."
            embedding = embedding_model.embed_query(test_text)
            
            # Check if we got a valid embedding (list of floats)
            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], float):
                logger.info("Gemini embedding model test successful")
                return True
            else:
                logger.error("Gemini embedding model test failed: Invalid embedding format")
                return False
                
        except Exception as e:
            logger.error(f"Gemini embedding model test failed: {e}")
            return False

# Global embedding manager instance
embedding_manager = EmbeddingManager()