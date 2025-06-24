"""Text chunking strategies for RAG document processing."""

import re
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.core.logging_config import get_logger

logger = get_logger(__name__)

class MarkdownAwareChunker:
    """Handles markdown-aware document chunking that preserves tables and structures."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the markdown-aware document chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize the text splitter with markdown-aware settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraphs and sections
                "\n# ",  # H1 headers
                "\n## ", # H2 headers
                "\n### ", # H3 headers
                "\n\n---\n\n",  # Horizontal rules
                "\n",    # Lines
                " ",     # Words
                ".",     # Sentences
                ",",     # Clauses
                ""       # Characters
            ],
            keep_separator=True,
            add_start_index=True
        )
        
        # Regex patterns for markdown structures
        self.table_pattern = re.compile(
            r'(\|.*\|.*\n)+(\|[-\s|:]+\|.*\n)(\|.*\|.*\n)*',
            re.MULTILINE
        )
        
        self.code_block_pattern = re.compile(
            r'```[\s\S]*?```',
            re.MULTILINE
        )
        
        logger.info(f"Markdown-aware chunker initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def extract_markdown_structures(self, content: str) -> Tuple[List[Tuple[int, int, str]], str]:
        """
        Extract markdown tables and code blocks, replacing them with placeholders.
        
        Args:
            content: Original markdown content
            
        Returns:
            Tuple of (structures_list, content_with_placeholders)
            where structures_list contains (start, end, type, content) tuples
        """
        structures = []
        
        # Find all tables
        for match in self.table_pattern.finditer(content):
            structures.append((
                match.start(),
                match.end(),
                "table",
                match.group()
            ))
        
        # Find all code blocks
        for match in self.code_block_pattern.finditer(content):
            structures.append((
                match.start(),
                match.end(),
                "code_block",
                match.group()
            ))
        
        # Sort by start position
        structures.sort(key=lambda x: x[0])
        
        # Replace structures with placeholders
        content_with_placeholders = content
        offset = 0
        
        for i, (start, end, struct_type, struct_content) in enumerate(structures):
            placeholder = f"\n\n__STRUCTURE_{i}_{struct_type.upper()}__\n\n"
            
            # Adjust positions based on previous replacements
            adjusted_start = start - offset
            adjusted_end = end - offset
            
            content_with_placeholders = (
                content_with_placeholders[:adjusted_start] +
                placeholder +
                content_with_placeholders[adjusted_end:]
            )
            
            # Update offset for next replacement
            offset += (end - start) - len(placeholder)
        
        return structures, content_with_placeholders
    
    def restore_structures(self, chunks: List[str], structures: List[Tuple[int, int, str, str]]) -> List[str]:
        """
        Restore markdown structures in chunks, keeping tables and code blocks intact.
        
        Args:
            chunks: List of text chunks with placeholders
            structures: List of original structures
            
        Returns:
            List of chunks with restored structures
        """
        restored_chunks = []
        
        for chunk in chunks:
            restored_chunk = chunk
            
            # Find placeholders in this chunk
            placeholder_pattern = re.compile(r'__STRUCTURE_(\d+)_(\w+)__')
            
            for match in placeholder_pattern.finditer(chunk):
                structure_index = int(match.group(1))
                
                if structure_index < len(structures):
                    original_structure = structures[structure_index][3]
                    restored_chunk = restored_chunk.replace(match.group(), original_structure)
            
            restored_chunks.append(restored_chunk)
        
        return restored_chunks
    
    def chunk_document(self, content: str, source_metadata: Dict[str, Any]) -> List[Document]:
        """
        Chunk a markdown document while preserving tables and code blocks.
        
        Args:
            content: The markdown content to chunk
            source_metadata: Metadata about the source document
            
        Returns:
            List of Document objects with chunked content and enhanced metadata
        """
        try:
            # Extract markdown structures (tables, code blocks) and replace with placeholders
            structures, content_with_placeholders = self.extract_markdown_structures(content)
            
            # Create a document object with placeholders
            doc = Document(
                page_content=content_with_placeholders,
                metadata=source_metadata
            )
            
            # Split the document into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            # Restore markdown structures in chunks
            chunk_contents = [chunk.page_content for chunk in chunks]
            restored_contents = self.restore_structures(chunk_contents, structures)
            
            # Create enhanced chunks with restored content
            enhanced_chunks = []
            for i, (chunk, restored_content) in enumerate(zip(chunks, restored_contents)):
                # Add chunk-specific metadata
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(restored_content),
                    "chunk_id": f"{source_metadata.get('source_id', 'unknown')}_{i}",
                    "has_table": "table" in restored_content.lower() and "|" in restored_content,
                    "has_code": "```" in restored_content
                })
                
                # Update the chunk content with restored structures
                chunk.page_content = restored_content
                enhanced_chunks.append(chunk)
            
            logger.info(f"Document chunked into {len(enhanced_chunks)} markdown-aware pieces")
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Error chunking markdown document: {e}")
            # Fallback to regular chunking if markdown processing fails
            return self._fallback_chunk(content, source_metadata)
    
    def _fallback_chunk(self, content: str, source_metadata: Dict[str, Any]) -> List[Document]:
        """Fallback chunking method if markdown-aware chunking fails."""
        try:
            doc = Document(page_content=content, metadata=source_metadata)
            chunks = self.text_splitter.split_documents([doc])
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content),
                    "chunk_id": f"{source_metadata.get('source_id', 'unknown')}_{i}"
                })
            
            logger.warning(f"Used fallback chunking for {len(chunks)} pieces")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in fallback chunking: {e}")
            raise
    
    def chunk_multiple_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Chunk multiple documents for batch processing.
        
        Args:
            documents: List of dictionaries with 'content' and 'metadata' keys
            
        Returns:
            List of chunked Document objects
        """
        all_chunks = []
        
        for doc_data in documents:
            content = doc_data.get('content', '')
            metadata = doc_data.get('metadata', {})
            
            if content.strip():  # Only process non-empty content
                chunks = self.chunk_document(content, metadata)
                all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks
    
    def get_chunk_preview(self, chunks: List[Document], max_chunks: int = 5) -> str:
        """
        Generate a preview of chunks for debugging/logging.
        
        Args:
            chunks: List of Document chunks
            max_chunks: Maximum number of chunks to include in preview
            
        Returns:
            String preview of chunks
        """
        preview = f"Document Chunks Preview ({len(chunks)} total chunks):\n"
        preview += "=" * 50 + "\n"
        
        for i, chunk in enumerate(chunks[:max_chunks]):
            has_table = chunk.metadata.get('has_table', False)
            has_code = chunk.metadata.get('has_code', False)
            
            preview += f"Chunk {i + 1}:\n"
            preview += f"  Length: {len(chunk.page_content)} characters\n"
            preview += f"  Has Table: {has_table}, Has Code: {has_code}\n"
            preview += f"  Metadata: {chunk.metadata}\n"
            preview += f"  Content preview: {chunk.page_content[:100]}...\n"
            preview += "-" * 30 + "\n"
        
        if len(chunks) > max_chunks:
            preview += f"... and {len(chunks) - max_chunks} more chunks\n"
        
        return preview

# Global chunker instance with optimized settings for markdown RAG
document_chunker = MarkdownAwareChunker(chunk_size=1000, chunk_overlap=200)