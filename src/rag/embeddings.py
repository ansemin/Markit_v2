"""Embedding model management for RAG functionality."""

import os
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from src.core.config import config
from src.core.logging_config import get_logger

logger = get_logger(__name__)

class EmbeddingManager:
    """Manages embedding models for document vectorization."""
    
    def __init__(self):
        self._embedding_model: Optional[OpenAIEmbeddings] = None
        
    def get_embedding_model(self) -> OpenAIEmbeddings:
        """Get or create the OpenAI embedding model."""
        if self._embedding_model is None:
            try:
                # Get OpenAI API key from config/environment
                openai_api_key = config.api.openai_api_key or os.getenv("OPENAI_API_KEY")
                
                if not openai_api_key:
                    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in environment variables.")
                
                self._embedding_model = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=openai_api_key,
                    chunk_size=1000,  # Process documents in chunks
                    max_retries=3,
                    timeout=30
                )
                
                logger.info("OpenAI embedding model initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embedding model: {e}")
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
                logger.info("Embedding model test successful")
                return True
            else:
                logger.error("Embedding model test failed: Invalid embedding format")
                return False
                
        except Exception as e:
            logger.error(f"Embedding model test failed: {e}")
            return False

# Global embedding manager instance
embedding_manager = EmbeddingManager()