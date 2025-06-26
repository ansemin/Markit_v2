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
        logger.info("Document ingestion service initialized")
    
    def create_file_hash(self, content: str) -> str:
        """Create a full SHA-256 hash for file content to avoid duplicates."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
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
            "ingestion_version": "1.0"
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
            
        return metadata
    
    def check_duplicate_in_vector_store(self, file_hash: str) -> bool:
        """Check if document with given file hash already exists in vector store."""
        try:
            existing_docs = vector_store_manager.get_vector_store()._collection.get(
                where={"file_hash": file_hash},
                limit=1
            )
            return len(existing_docs.get('ids', [])) > 0
        except Exception as e:
            logger.error(f"Error checking for duplicates: {e}")
            return False
    
    def delete_existing_document(self, file_hash: str) -> bool:
        """Delete existing document with given file hash from vector store."""
        try:
            vector_store_manager.get_vector_store()._collection.delete(
                where={"file_hash": file_hash}
            )
            logger.info(f"Deleted existing document with hash: {file_hash}")
            return True
        except Exception as e:
            logger.error(f"Error deleting existing document: {e}")
            return False
    
    def ingest_text_content(self, 
                           text_content: str, 
                           content_type: str = "markdown",
                           source_path: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           original_file_content: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Ingest text content (markdown or LaTeX) into the RAG system.
        
        Args:
            text_content: The text content to ingest (markdown or LaTeX)
            content_type: Type of content ("markdown" or "latex")
            source_path: Optional source path/filename
            metadata: Optional additional metadata
            original_file_content: Original file content for hash calculation
            
        Returns:
            Tuple of (success, message, ingestion_stats)
        """
        try:
            if not text_content or not text_content.strip():
                return False, "No content provided for ingestion", {}
            
            # Create file hash using original content if available, otherwise use text content
            file_content_for_hash = original_file_content or text_content
            file_hash = self.create_file_hash(file_content_for_hash)
            
            # Check for duplicates in vector store
            is_duplicate = self.check_duplicate_in_vector_store(file_hash)
            replacement_mode = False
            
            if is_duplicate:
                logger.info(f"Document with hash {file_hash} already exists, replacing...")
                # Delete existing document
                if self.delete_existing_document(file_hash):
                    replacement_mode = True
                else:
                    return False, "Failed to replace existing document", {"status": "error"}
            
            # Prepare document metadata with file hash
            doc_metadata = self.prepare_document_metadata(
                source_path=source_path,
                doc_type=content_type,  # Use content_type instead of hardcoded "markdown"
                additional_metadata=metadata
            )
            doc_metadata["file_hash"] = file_hash
            doc_metadata["content_length"] = len(text_content)
            doc_metadata["upload_timestamp"] = datetime.now().isoformat()
            
            # Chunk the document using text-aware chunking
            logger.info(f"Chunking {content_type} document: {file_hash}")
            chunks = document_chunker.chunk_document(text_content, doc_metadata)
            
            if not chunks:
                return False, "Failed to create document chunks", {}
            
            # Add chunks to vector store
            logger.info(f"Adding {len(chunks)} chunks to vector store")
            doc_ids = vector_store_manager.add_documents(chunks)
            
            if not doc_ids:
                return False, "Failed to add documents to vector store", {}
            
            # Prepare ingestion statistics
            ingestion_stats = {
                "status": "success",
                "file_hash": file_hash,
                "total_chunks": len(chunks),
                "document_ids": doc_ids,
                "content_length": len(text_content),
                "has_tables": any(chunk.metadata.get("has_table", False) for chunk in chunks),
                "has_code": any(chunk.metadata.get("has_code", False) for chunk in chunks),
                "processed_at": datetime.now().isoformat(),
                "replacement_mode": replacement_mode
            }
            
            action = "Updated existing" if replacement_mode else "Successfully ingested"
            success_msg = f"{action} document with {len(chunks)} chunks"
            logger.info(f"{success_msg}: {file_hash}")
            
            return True, success_msg, ingestion_stats
            
        except Exception as e:
            error_msg = f"Error during document ingestion: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {"status": "error", "error": str(e)}
    
    def ingest_markdown_content(self, 
                              markdown_content: str, 
                              source_path: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None,
                              original_file_content: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Backward compatibility method for ingesting markdown content.
        """
        return self.ingest_text_content(
            text_content=markdown_content,
            content_type="markdown",
            source_path=source_path,
            metadata=metadata,
            original_file_content=original_file_content
        )
    
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
            original_file_content = conversion_result.get("original_file_content")
            
            additional_metadata = {
                "original_filename": original_filename,
                "conversion_method": conversion_method,
                "file_size": conversion_result.get("file_size", 0),
                "conversion_time": conversion_result.get("conversion_time", 0)
            }
            
            # Determine content type based on conversion method
            content_type = "latex" if "GOT-OCR" in conversion_method else "markdown"
            
            # Ingest the content with original file content for proper hashing
            return self.ingest_text_content(
                text_content=markdown_content,
                content_type=content_type,
                source_path=original_filename,
                metadata=additional_metadata,
                original_file_content=original_file_content
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
            "processed_documents": 0,  # Will be updated from vector store
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