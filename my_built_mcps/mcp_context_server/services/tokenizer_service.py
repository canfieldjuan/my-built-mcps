"""
services/tokenizer_service.py

Tokenization service using tiktoken.
"""
from typing import List
import tiktoken

from ..config.settings import settings
from ..interfaces import AbstractTokenizer


class TiktokenService(AbstractTokenizer):
    """Tiktoken-based tokenization service."""
    
    def __init__(self, encoding: str = None):
        self.encoding_name = encoding or settings.TOKENIZER_ENCODING
        self._encoding = None
    
    @property
    def encoding(self):
        """Lazy load encoding."""
        if self._encoding is None:
            self._encoding = tiktoken.get_encoding(self.encoding_name)
        return self._encoding
    
    def encode(self, text: str) -> List[int]:
        """Encode text into tokens."""
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to text."""
        return self.encoding.decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        return len(self.encode(text))


# Default tokenizer instance
tokenizer = TiktokenService()