"""Threading utilities for UI components."""

import threading
import time
import logging

from src.core.converter import is_conversion_in_progress
from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Global variable to track cancellation state
conversion_cancelled = threading.Event()


def monitor_cancellation():
    """Background thread to monitor cancellation and update UI if needed"""
    logger.info("Starting cancellation monitor thread")
    while is_conversion_in_progress():
        if conversion_cancelled.is_set():
            logger.info("Cancellation detected by monitor thread")
        time.sleep(0.1)  # Check every 100ms
    logger.info("Cancellation monitor thread ending")


def get_cancellation_event():
    """Get the global cancellation event."""
    return conversion_cancelled


def reset_cancellation():
    """Reset the cancellation event."""
    conversion_cancelled.clear()


def set_cancellation():
    """Set the cancellation event."""
    conversion_cancelled.set()