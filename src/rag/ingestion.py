"""Document ingestion pipeline for RAG functionality."""

import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document
from src.rag.chunking import document_chunker
from src.rag.vector_store import vector_store_manager
from src.rag.embeddings import embedding_manager
from src.core.logging_config import get_logger

logger = get_logger(__name__)

class DocumentIngestionService:
    """Service for ingesting documents into the RAG system."""
    
    def __init__(self):
        """Initialize the document ingestion service."""
        self.processed_documents = set()  # Track processed document hashes
        logger.info("Document ingestion service initialized")
    
    def create_document_hash(self, content: str) -> str:
        """Create a hash for document content to avoid duplicates."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def prepare_document_metadata(self, 
                                source_path: Optional[str] = None, 
                                doc_type: str = "markdown",
                                additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare metadata for a document.
        
        Args:
            source_path: Original document path
            doc_type: Type of document (markdown, pdf, etc.)
            additional_metadata: Additional metadata to include
            
        Returns:
            Dictionary with document metadata
        """
        metadata = {
            "source": source_path or "user_upload",
            "doc_type": doc_type,
            "processed_at": datetime.now().isoformat(),
            "source_id": self.create_document_hash(source_path or ""),
            "ingestion_version": "1.0"
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
            
        return metadata
    
    def ingest_markdown_content(self, 
                              markdown_content: str, 
                              source_path: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Ingest markdown content into the RAG system.
        
        Args:
            markdown_content: The markdown content to ingest
            source_path: Optional source path/filename
            metadata: Optional additional metadata
            
        Returns:
            Tuple of (success, message, ingestion_stats)
        """
        try:
            if not markdown_content or not markdown_content.strip():
                return False, "No content provided for ingestion", {}
            
            # Create document hash to check for duplicates
            content_hash = self.create_document_hash(markdown_content)
            
            if content_hash in self.processed_documents:
                logger.info(f"Document already processed: {content_hash}")
                return True, "Document already exists in the system", {"status": "duplicate"}
            
            # Prepare document metadata
            doc_metadata = self.prepare_document_metadata(
                source_path=source_path,
                doc_type="markdown",
                additional_metadata=metadata
            )
            doc_metadata["content_hash"] = content_hash
            doc_metadata["content_length"] = len(markdown_content)
            
            # Chunk the document using markdown-aware chunking
            logger.info(f"Chunking document: {content_hash}")
            chunks = document_chunker.chunk_document(markdown_content, doc_metadata)
            
            if not chunks:
                return False, "Failed to create document chunks", {}
            
            # Add chunks to vector store
            logger.info(f"Adding {len(chunks)} chunks to vector store")
            doc_ids = vector_store_manager.add_documents(chunks)
            
            if not doc_ids:
                return False, "Failed to add documents to vector store", {}
            
            # Mark document as processed
            self.processed_documents.add(content_hash)
            
            # Prepare ingestion statistics
            ingestion_stats = {
                "status": "success",
                "content_hash": content_hash,
                "total_chunks": len(chunks),
                "document_ids": doc_ids,
                "content_length": len(markdown_content),
                "has_tables": any(chunk.metadata.get("has_table", False) for chunk in chunks),
                "has_code": any(chunk.metadata.get("has_code", False) for chunk in chunks),
                "processed_at": datetime.now().isoformat()
            }
            
            success_msg = f"Successfully ingested document with {len(chunks)} chunks"
            logger.info(f"{success_msg}: {content_hash}")
            
            return True, success_msg, ingestion_stats
            
        except Exception as e:
            error_msg = f"Error during document ingestion: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {"status": "error", "error": str(e)}
    
    def ingest_from_conversion_result(self, conversion_result: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Ingest a document from Markit conversion result.
        
        Args:
            conversion_result: Dictionary containing conversion results from Markit
            
        Returns:
            Tuple of (success, message, ingestion_stats)
        """
        try:
            # Extract markdown content from conversion result
            markdown_content = conversion_result.get("markdown_content", "")
            
            if not markdown_content:
                return False, "No markdown content found in conversion result", {}
            
            # Extract metadata from conversion result
            original_filename = conversion_result.get("original_filename", "unknown")
            conversion_method = conversion_result.get("conversion_method", "unknown")
            
            additional_metadata = {
                "original_filename": original_filename,
                "conversion_method": conversion_method,
                "file_size": conversion_result.get("file_size", 0),
                "conversion_time": conversion_result.get("conversion_time", 0)
            }
            
            # Ingest the markdown content
            return self.ingest_markdown_content(
                markdown_content=markdown_content,
                source_path=original_filename,
                metadata=additional_metadata
            )
            
        except Exception as e:
            error_msg = f"Error ingesting from conversion result: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {"status": "error", "error": str(e)}
    
    def get_ingestion_status(self) -> Dict[str, Any]:
        """
        Get current ingestion system status.
        
        Returns:
            Dictionary with system status information
        """
        status = {
            "processed_documents": len(self.processed_documents),
            "embedding_model_available": False,
            "vector_store_available": False,
            "system_ready": False
        }
        
        try:
            # Check embedding model
            status["embedding_model_available"] = embedding_manager.test_embedding_model()
            
            # Check vector store
            collection_info = vector_store_manager.get_collection_info()
            status["vector_store_available"] = "error" not in collection_info
            status["total_documents_in_store"] = collection_info.get("document_count", 0)
            
            # System is ready if both components are available
            status["system_ready"] = (
                status["embedding_model_available"] and 
                status["vector_store_available"]
            )
            
        except Exception as e:
            logger.error(f"Error checking ingestion status: {e}")
            status["error"] = str(e)
        
        return status
    
    def clear_processed_documents(self) -> None:
        """Clear the set of processed documents."""
        self.processed_documents.clear()
        logger.info("Cleared processed documents cache")
    
    def test_ingestion_pipeline(self) -> Dict[str, Any]:
        """
        Test the complete ingestion pipeline with sample content.
        
        Returns:
            Dictionary with test results
        """
        test_results = {
            "pipeline_test": False,
            "chunking_test": False,
            "embedding_test": False,
            "vector_store_test": False,
            "errors": []
        }
        
        try:
            # Test with sample markdown content
            test_content = """# Test Document

This is a test document for the RAG ingestion pipeline.

## Features

- Document chunking
- Embedding generation
- Vector store integration

## Sample Table

| Feature | Status | Priority |
|---------|--------|----------|
| Chunking | ✅ | High |
| Embeddings | ✅ | High |
| Vector Store | ✅ | Medium |

```python
# Sample code block
def test_function():
    return "Hello, RAG!"
```

This document contains various markdown elements to test the ingestion pipeline.
"""
            
            # Test chunking
            metadata = self.prepare_document_metadata(
                source_path="test_document.md",
                doc_type="markdown"
            )
            
            chunks = document_chunker.chunk_document(test_content, metadata)
            test_results["chunking_test"] = len(chunks) > 0
            
            if not test_results["chunking_test"]:
                test_results["errors"].append("Chunking test failed: No chunks created")
                return test_results
            
            # Test embedding
            test_results["embedding_test"] = embedding_manager.test_embedding_model()
            
            if not test_results["embedding_test"]:
                test_results["errors"].append("Embedding test failed")
                return test_results
            
            # Test vector store (add and retrieve)
            doc_ids = vector_store_manager.add_documents(chunks[:1])  # Test with one chunk
            test_results["vector_store_test"] = len(doc_ids) > 0
            
            if test_results["vector_store_test"]:
                # Test retrieval
                search_results = vector_store_manager.similarity_search("test document", k=1)
                test_results["vector_store_test"] = len(search_results) > 0
            
            if not test_results["vector_store_test"]:
                test_results["errors"].append("Vector store test failed")
                return test_results
            
            # Overall pipeline test
            test_results["pipeline_test"] = (
                test_results["chunking_test"] and 
                test_results["embedding_test"] and 
                test_results["vector_store_test"]
            )
            
            logger.info(f"Ingestion pipeline test completed: {test_results['pipeline_test']}")
            
        except Exception as e:
            error_msg = f"Pipeline test error: {str(e)}"
            test_results["errors"].append(error_msg)
            logger.error(error_msg)
        
        return test_results

# Global document ingestion service instance
document_ingestion_service = DocumentIngestionService()