"""Vector store management using Chroma for document storage and retrieval."""

import os
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from src.rag.embeddings import embedding_manager
from src.core.config import config
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class LimitedEnsembleRetriever:
    """Simple wrapper around EnsembleRetriever that limits total results to k."""
    
    def __init__(self, ensemble_retriever: EnsembleRetriever, k: int):
        self.ensemble_retriever = ensemble_retriever
        self.k = k
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents, limited to k results."""
        # Use invoke method instead of deprecated get_relevant_documents
        docs = self.ensemble_retriever.invoke(query)
        # Limit to k results
        return docs[:self.k]
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        docs = await self.ensemble_retriever.ainvoke(query)
        return docs[:self.k]
    
    def invoke(self, input_data, config=None, **kwargs):
        """Compatibility method for invoke interface."""
        if isinstance(input_data, str):
            return self.get_relevant_documents(input_data)
        return self.get_relevant_documents(input_data)


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
        self._documents_cache: List[Document] = []  # Cache documents for BM25 retriever
        self._bm25_retriever: Optional[BM25Retriever] = None
        
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
            
            # Update documents cache for BM25 retriever
            self._documents_cache.extend(documents)
            # Reset BM25 retriever to force rebuild with new documents
            self._bm25_retriever = None
            
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
    
    def get_bm25_retriever(self, k: int = 4) -> BM25Retriever:
        """
        Get or create a BM25 retriever for keyword-based search.
        
        Args:
            k: Number of documents to return
            
        Returns:
            BM25Retriever object
        """
        try:
            if self._bm25_retriever is None or not self._documents_cache:
                if not self._documents_cache:
                    # Try to load documents from the vector store
                    vector_store = self.get_vector_store()
                    collection = vector_store._collection
                    all_docs = collection.get()
                    
                    if all_docs and all_docs.get('documents') and all_docs.get('metadatas'):
                        # Reconstruct documents from vector store
                        self._documents_cache = [
                            Document(page_content=content, metadata=metadata)
                            for content, metadata in zip(all_docs['documents'], all_docs['metadatas'])
                        ]
                
                if self._documents_cache:
                    self._bm25_retriever = BM25Retriever.from_documents(
                        documents=self._documents_cache,
                        k=k
                    )
                    logger.info(f"Created BM25 retriever with {len(self._documents_cache)} documents")
                else:
                    logger.warning("No documents available for BM25 retriever")
                    # Create empty retriever
                    self._bm25_retriever = BM25Retriever.from_documents(
                        documents=[Document(page_content="", metadata={})],
                        k=k
                    )
            
            # Update k if different
            if hasattr(self._bm25_retriever, 'k'):
                self._bm25_retriever.k = k
            
            return self._bm25_retriever
            
        except Exception as e:
            logger.error(f"Error creating BM25 retriever: {e}")
            raise
    
    def get_hybrid_retriever(self, 
                           k: int = 4, 
                           semantic_weight: float = 0.7, 
                           keyword_weight: float = 0.3,
                           search_type: str = "similarity",
                           search_kwargs: Optional[Dict[str, Any]] = None) -> LimitedEnsembleRetriever:
        """
        Get a hybrid retriever that combines semantic (vector) and keyword (BM25) search.
        
        Args:
            k: Number of documents to return (exactly k results will be returned)
            semantic_weight: Weight for semantic search (0.0 to 1.0)
            keyword_weight: Weight for keyword search (0.0 to 1.0)
            search_type: Type of semantic search ("similarity", "mmr", "similarity_score_threshold")
            search_kwargs: Additional search parameters for semantic retriever
            
        Returns:
            LimitedEnsembleRetriever object that returns exactly k results
        """
        try:
            # Normalize weights
            total_weight = semantic_weight + keyword_weight
            if total_weight == 0:
                semantic_weight, keyword_weight = 0.7, 0.3
            else:
                semantic_weight = semantic_weight / total_weight
                keyword_weight = keyword_weight / total_weight
            
            # Get semantic retriever
            if search_kwargs is None:
                search_kwargs = {"k": k}
            else:
                search_kwargs = search_kwargs.copy()
                search_kwargs["k"] = k
            
            semantic_retriever = self.get_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            
            # Get BM25 retriever
            keyword_retriever = self.get_bm25_retriever(k=k)
            
            # Create ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, keyword_retriever],
                weights=[semantic_weight, keyword_weight]
            )
            
            # Wrap with LimitedEnsembleRetriever to ensure exactly k results
            limited_retriever = LimitedEnsembleRetriever(ensemble_retriever, k)
            
            logger.info(f"Created hybrid retriever with weights: semantic={semantic_weight:.2f}, keyword={keyword_weight:.2f}, limited to {k} results")
            return limited_retriever
            
        except Exception as e:
            logger.error(f"Error creating hybrid retriever: {e}")
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
                "embedding_model": config.rag.embedding_model
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
    
    def reset_vector_store(self) -> bool:
        """
        Reset the vector store completely.
        This will clear all documents and recreate the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Resetting vector store...")
            
            # Clear all documents and reset the vector store
            success = self.clear_all_documents()
            
            if success:
                # Also delete the collection to ensure clean state
                if self._vector_store is not None:
                    self._vector_store.delete_collection()
                    self._vector_store = None
                
                logger.info("Vector store reset successfully")
                return True
            else:
                logger.error("Failed to reset vector store")
                return False
                
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")
            return False
    
    def clear_all_documents(self) -> bool:
        """
        Clear all documents from the vector store collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            vector_store = self.get_vector_store()
            
            # Get all document IDs first
            collection = vector_store._collection
            all_docs = collection.get()
            
            if not all_docs or not all_docs.get('ids'):
                logger.info("No documents found in vector store to clear")
                return True
            
            # Delete all documents using their IDs
            collection.delete(ids=all_docs['ids'])
            
            # Reset the vector store instance to ensure clean state
            self._vector_store = None
            
            # Clear documents cache and BM25 retriever
            self._documents_cache.clear()
            self._bm25_retriever = None
            
            logger.info(f"Successfully cleared {len(all_docs['ids'])} documents from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing all documents: {e}")
            return False

# Global vector store manager instance
vector_store_manager = VectorStoreManager()