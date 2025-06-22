"""
Custom exception classes for the Markit application.
"""


class MarkitError(Exception):
    """Base exception class for all Markit-related errors."""
    pass


class ConfigurationError(MarkitError):
    """Raised when there's a configuration-related error."""
    pass


class ParserError(MarkitError):
    """Base exception for parser-related errors."""
    pass


class ParserNotFoundError(ParserError):
    """Raised when a requested parser is not available."""
    pass


class ParserInitializationError(ParserError):
    """Raised when a parser fails to initialize properly."""
    pass


class DocumentProcessingError(ParserError):
    """Raised when document processing fails."""
    pass


class UnsupportedFileTypeError(ParserError):
    """Raised when trying to process an unsupported file type."""
    pass


class APIError(MarkitError):
    """Base exception for API-related errors."""
    pass


class APIKeyMissingError(APIError):
    """Raised when required API key is missing."""
    pass


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass


class APIQuotaExceededError(APIError):
    """Raised when API quota is exceeded."""
    pass


class FileError(MarkitError):
    """Base exception for file-related errors."""
    pass


class FileSizeLimitError(FileError):
    """Raised when file size exceeds the allowed limit."""
    pass


class FileNotFoundError(FileError):
    """Raised when a required file is not found."""
    pass


class ConversionError(MarkitError):
    """Raised when document conversion fails."""
    pass


class ValidationError(MarkitError):
    """Raised when input validation fails."""
    pass