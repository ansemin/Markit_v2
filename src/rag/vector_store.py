"""Vector store management using Chroma for document storage and retrieval."""

import os
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from src.rag.embeddings import embedding_manager
from src.core.config import config
from src.core.logging_config import get_logger

logger = get_logger(__name__)

class VectorStoreManager:
    """Manages Chroma vector store for document storage and retrieval."""
    
    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "markit_documents"):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection in the vector store
        """
        self.collection_name = collection_name
        
        # Set default persist directory
        if persist_directory is None:
            persist_directory = config.rag.vector_store_path
        
        self.persist_directory = str(Path(persist_directory).resolve())
        
        # Ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self._vector_store: Optional[Chroma] = None
        
        logger.info(f"VectorStoreManager initialized with persist_directory={self.persist_directory}")
    
    def get_vector_store(self) -> Chroma:
        """Get or create the Chroma vector store."""
        if self._vector_store is None:
            try:
                embedding_model = embedding_manager.get_embedding_model()
                
                self._vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=embedding_model,
                    persist_directory=self.persist_directory,
                    collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                
                logger.info(f"Vector store initialized with collection '{self.collection_name}'")
                
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                raise
        
        return self._vector_store
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs that were added
        """
        try:
            if not documents:
                logger.warning("No documents provided to add to vector store")
                return []
            
            vector_store = self.get_vector_store()
            
            # Generate unique IDs for documents
            doc_ids = [f"doc_{i}_{hash(doc.page_content)}" for i, doc in enumerate(documents)]
            
            # Add documents to the vector store
            added_ids = vector_store.add_documents(documents=documents, ids=doc_ids)
            
            logger.info(f"Added {len(added_ids)} documents to vector store")
            return added_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 4, score_threshold: Optional[float] = None) -> List[Document]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query
            k: Number of documents to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar documents
        """
        try:
            vector_store = self.get_vector_store()
            
            if score_threshold is not None:
                # Use similarity search with score threshold
                docs_with_scores = vector_store.similarity_search_with_relevance_scores(
                    query=query,
                    k=k,
                    score_threshold=score_threshold
                )
                documents = [doc for doc, score in docs_with_scores]
            else:
                # Regular similarity search
                documents = vector_store.similarity_search(query=query, k=k)
            
            logger.info(f"Found {len(documents)} similar documents for query: '{query[:50]}...'")
            return documents
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict[str, Any]] = None) -> VectorStoreRetriever:
        """
        Get a retriever for the vector store.
        
        Args:
            search_type: Type of search ("similarity", "mmr", "similarity_score_threshold")
            search_kwargs: Additional search parameters
            
        Returns:
            VectorStoreRetriever object
        """
        try:
            vector_store = self.get_vector_store()
            
            if search_kwargs is None:
                search_kwargs = {"k": 4}
            
            retriever = vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            
            logger.info(f"Created retriever with search_type='{search_type}' and kwargs={search_kwargs}")
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            vector_store = self.get_vector_store()
            
            # Get collection count
            count = vector_store._collection.count()
            
            info = {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "document_count": count,
                "embedding_model": "text-embedding-3-small"
            }
            
            logger.info(f"Collection info: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """
        Delete the current collection and reset the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self._vector_store is not None:
                self._vector_store.delete_collection()
                self._vector_store = None
                
            logger.info(f"Deleted collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def search_with_metadata_filter(self, query: str, metadata_filter: Dict[str, Any], k: int = 4) -> List[Document]:
        """
        Search documents with metadata filtering.
        
        Args:
            query: Search query
            metadata_filter: Metadata filter conditions
            k: Number of documents to return
            
        Returns:
            List of filtered documents
        """
        try:
            vector_store = self.get_vector_store()
            
            documents = vector_store.similarity_search(
                query=query,
                k=k,
                filter=metadata_filter
            )
            
            logger.info(f"Found {len(documents)} documents with metadata filter: {metadata_filter}")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching with metadata filter: {e}")
            return []

# Global vector store manager instance
vector_store_manager = VectorStoreManager()