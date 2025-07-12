"""
config/settings.py

Configuration settings for MCP Context Server.
All environment variables and configuration constants.
"""
import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""
    
    # Database
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY")
    
    # Cache
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # AI Services
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "claude-3-haiku-20240307"
    TOKENIZER_ENCODING: str = "cl100k_base"
    
    # Memory Management
    SHORT_TERM_LIMIT: int = 4096
    SUMMARIZE_THRESHOLD: int = 2048
    CACHE_TTL: int = 3600
    
    # Session Management
    MAX_SESSIONS: int = 1000
    SESSION_TTL_HOURS: int = 24
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Feature Flags
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    @classmethod
    def validate(cls) -> None:
        """Validate required settings are present."""
        required = ["SUPABASE_URL", "SUPABASE_KEY", "ANTHROPIC_API_KEY"]
        missing = [var for var in required if not getattr(cls, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


# Singleton instance
settings = Settings()