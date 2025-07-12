"""
core/exceptions.py

Custom exceptions for the MCP Context Server.
"""


class MCPError(Exception):
    """Base exception for MCP Context Server."""
    pass


class SessionError(MCPError):
    """Raised when there's an issue with session management."""
    pass


class ContextError(MCPError):
    """Raised when there's an issue with context management."""
    pass


class StorageError(MCPError):
    """Raised when there's an issue with storage operations."""
    pass


class CacheError(MCPError):
    """Raised when there's an issue with cache operations."""
    pass


class LLMError(MCPError):
    """Raised when there's an issue with LLM operations."""
    pass


class EmbeddingError(MCPError):
    """Raised when there's an issue with embedding operations."""
    pass


class ValidationError(MCPError):
    """Raised when validation fails."""
    pass