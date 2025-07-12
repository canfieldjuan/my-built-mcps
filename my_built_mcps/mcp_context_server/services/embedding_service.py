"""
services/embedding_service.py

Embedding generation service using Sentence Transformers.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config.settings import settings
from ..interfaces import AbstractEmbedder


class SentenceTransformerEmbedder(AbstractEmbedder):
    """Sentence Transformer based embedding service."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts)
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced."""
        # For all-MiniLM-L6-v2, this is 384
        return 384
    
    def normalize_embedding(self, embedding: np.ndarray) -> List[float]:
        """Normalize embedding for pgvector."""
        # Ensure float32 and normalize
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()


# Default embedder instance
embedder = SentenceTransformerEmbedder()