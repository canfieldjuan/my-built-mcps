"""
interfaces/__init__.py (and individual interface files)

Abstract base classes defining interfaces for pluggable components.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class AbstractSummarizer(ABC):
    """Interface for text summarization services."""
    
    @abstractmethod
    async def summarize(self, content: str, max_length: int = 500) -> str:
        """Generate a summary of the given content."""
        pass
    
    @abstractmethod
    async def extract_topics(self, content: str, max_topics: int = 5) -> List[str]:
        """Extract key topics from the content."""
        pass


class AbstractEmbedder(ABC):
    """Interface for embedding generation services."""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced."""
        pass


class AbstractTokenizer(ABC):
    """Interface for text tokenization services."""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text into tokens."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to text."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        pass


class AbstractVectorStore(ABC):
    """Interface for vector similarity search."""
    
    @abstractmethod
    async def search(
        self, 
        embedding: List[float], 
        k: int = 10, 
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def upsert(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Insert or update a vector."""
        pass


class AbstractCache(ABC):
    """Interface for caching services."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate all keys matching pattern."""
        pass


class AbstractContextStorage(ABC):
    """Interface for context node storage."""
    
    @abstractmethod
    async def save_node(self, node: Dict[str, Any]) -> str:
        """Save a context node and return its ID."""
        pass
    
    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a context node by ID."""
        pass
    
    @abstractmethod
    async def update_node(self, node_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing node."""
        pass
    
    @abstractmethod
    async def query_nodes(
        self,
        filters: Dict[str, Any],
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query nodes with filters."""
        pass