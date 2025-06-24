"""
Data clearing service for both local and Hugging Face Space environments.
Provides functionality to clear vector store and chat history data.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List
from src.core.config import config
from src.core.logging_config import get_logger
from src.rag.vector_store import vector_store_manager

logger = get_logger(__name__)


class DataClearingService:
    """Service for clearing all RAG-related data across different environments."""
    
    def __init__(self):
        """Initialize the data clearing service."""
        self.is_hf_space = bool(os.getenv("SPACE_ID"))
        logger.info(f"DataClearingService initialized (HF Space: {self.is_hf_space})")
    
    def get_data_paths(self) -> Tuple[str, str]:
        """
        Get the correct data paths for current environment.
        
        Returns:
            Tuple of (vector_store_path, chat_history_path)
        """
        vector_store_path = config.rag.vector_store_path
        chat_history_path = config.rag.chat_history_path
        
        logger.info(f"Data paths - Vector store: {vector_store_path}, Chat history: {chat_history_path}")
        return vector_store_path, chat_history_path
    
    def clear_vector_store(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Clear all documents from the vector store.
        
        Returns:
            Tuple of (success, message, stats)
        """
        try:
            # Get initial document count
            collection_info = vector_store_manager.get_collection_info()
            initial_count = collection_info.get("document_count", 0)
            
            if initial_count == 0:
                return True, "Vector store is already empty", {"cleared_documents": 0}
            
            # Clear the collection using the vector store manager's method
            success = vector_store_manager.clear_all_documents()
            
            if not success:
                return False, "Failed to clear vector store", {"error": "clear_all_documents returned False"}
            
            logger.info(f"Cleared {initial_count} documents from vector store")
            
            return True, f"Successfully cleared {initial_count} documents from vector store", {
                "cleared_documents": initial_count,
                "collection_name": collection_info.get("collection_name", "unknown")
            }
            
        except Exception as e:
            error_msg = f"Error clearing vector store: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {"error": str(e)}
    
    def clear_chat_history(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Clear all chat history files.
        
        Returns:
            Tuple of (success, message, stats)
        """
        try:
            _, chat_history_path = self.get_data_paths()
            chat_dir = Path(chat_history_path)
            
            if not chat_dir.exists():
                return True, "Chat history directory doesn't exist", {"cleared_files": 0}
            
            # Count files before deletion
            files_to_clear = list(chat_dir.rglob("*"))
            file_count = len([f for f in files_to_clear if f.is_file()])
            
            if file_count == 0:
                return True, "Chat history is already empty", {"cleared_files": 0}
            
            # Clear all contents of the chat history directory
            for item in chat_dir.iterdir():
                if item.is_file():
                    item.unlink()
                    logger.debug(f"Removed file: {item}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    logger.debug(f"Removed directory: {item}")
            
            logger.info(f"Cleared {file_count} files from chat history")
            
            return True, f"Successfully cleared {file_count} files from chat history", {
                "cleared_files": file_count,
                "chat_history_path": str(chat_dir)
            }
            
        except Exception as e:
            error_msg = f"Error clearing chat history: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {"error": str(e)}
    
    def clear_directory_contents(self, directory_path: str) -> Tuple[bool, str, int]:
        """
        Clear all contents of a specific directory.
        
        Args:
            directory_path: Path to directory to clear
            
        Returns:
            Tuple of (success, message, items_cleared)
        """
        try:
            dir_path = Path(directory_path)
            
            if not dir_path.exists():
                return True, f"Directory doesn't exist: {directory_path}", 0
            
            items_cleared = 0
            for item in dir_path.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        items_cleared += 1
                        logger.debug(f"Removed file: {item}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        items_cleared += 1
                        logger.debug(f"Removed directory: {item}")
                except Exception as e:
                    logger.warning(f"Failed to remove {item}: {e}")
            
            return True, f"Cleared {items_cleared} items from {directory_path}", items_cleared
            
        except Exception as e:
            error_msg = f"Error clearing directory {directory_path}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, 0
    
    def clear_all_data(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Clear all RAG-related data (vector store + chat history).
        
        Returns:
            Tuple of (success, message, combined_stats)
        """
        logger.info("Starting complete data clearing operation")
        
        combined_stats = {
            "vector_store": {},
            "chat_history": {},
            "total_cleared_documents": 0,
            "total_cleared_files": 0,
            "environment": "hf_space" if self.is_hf_space else "local",
            "errors": []
        }
        
        # Clear vector store
        vs_success, vs_message, vs_stats = self.clear_vector_store()
        combined_stats["vector_store"] = {
            "success": vs_success,
            "message": vs_message,
            **vs_stats
        }
        
        if not vs_success:
            combined_stats["errors"].append(f"Vector store: {vs_message}")
        else:
            combined_stats["total_cleared_documents"] = vs_stats.get("cleared_documents", 0)
        
        # Clear chat history
        ch_success, ch_message, ch_stats = self.clear_chat_history()
        combined_stats["chat_history"] = {
            "success": ch_success,
            "message": ch_message,
            **ch_stats
        }
        
        if not ch_success:
            combined_stats["errors"].append(f"Chat history: {ch_message}")
        else:
            combined_stats["total_cleared_files"] = ch_stats.get("cleared_files", 0)
        
        # Overall success
        overall_success = vs_success and ch_success
        
        if overall_success:
            total_items = combined_stats["total_cleared_documents"] + combined_stats["total_cleared_files"]
            if total_items == 0:
                overall_message = "All data was already clear"
            else:
                overall_message = f"Successfully cleared all data: {combined_stats['total_cleared_documents']} documents, {combined_stats['total_cleared_files']} files"
        else:
            overall_message = f"Data clearing completed with errors: {'; '.join(combined_stats['errors'])}"
        
        logger.info(f"Data clearing operation completed: {overall_message}")
        
        return overall_success, overall_message, combined_stats
    
    def get_data_status(self) -> Dict[str, Any]:
        """
        Get current status of data directories and vector store.
        
        Returns:
            Dictionary with data status information
        """
        try:
            vector_store_path, chat_history_path = self.get_data_paths()
            
            # Vector store status
            collection_info = vector_store_manager.get_collection_info()
            vs_document_count = collection_info.get("document_count", 0)
            
            # Chat history status
            chat_dir = Path(chat_history_path)
            ch_file_count = 0
            if chat_dir.exists():
                ch_file_count = len([f for f in chat_dir.rglob("*") if f.is_file()])
            
            # Directory status
            vs_dir = Path(vector_store_path)
            vs_exists = vs_dir.exists()
            ch_exists = chat_dir.exists()
            
            status = {
                "environment": "hf_space" if self.is_hf_space else "local",
                "vector_store": {
                    "path": vector_store_path,
                    "exists": vs_exists,
                    "document_count": vs_document_count,
                    "collection_name": collection_info.get("collection_name", "unknown")
                },
                "chat_history": {
                    "path": chat_history_path,
                    "exists": ch_exists,
                    "file_count": ch_file_count
                },
                "total_data_items": vs_document_count + ch_file_count,
                "has_data": vs_document_count > 0 or ch_file_count > 0
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting data status: {e}")
            return {
                "error": str(e),
                "environment": "hf_space" if self.is_hf_space else "local"
            }


# Global data clearing service instance
data_clearing_service = DataClearingService()