"""
models/enums.py

Enumerations used throughout the application.
"""
from enum import Enum


class MemoryTier(str, Enum):
    """Memory tier levels for context management."""
    SHORT_TERM = "short_term"
    MID_TERM = "mid_term"
    LONG_TERM = "long_term"
    META = "meta"