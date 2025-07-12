"""
repositories/database.py

Database connection and initialization utilities.
"""
import asyncio
import logging
from typing import Optional
from supabase import create_client, Client

from ..config.settings import settings
from .schema import SQL_COMMANDS

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and initialization."""
    
    def __init__(self):
        self._client: Optional[Client] = None
    
    @property
    def client(self) -> Client:
        """Get or create Supabase client."""
        if self._client is None:
            self._client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        return self._client
    
    async def initialize_tables(self) -> None:
        """Initialize database tables if needed."""
        for sql in SQL_COMMANDS:
            try:
                # For dev environments - execute the SQL
                await asyncio.to_thread(
                    self.client.rpc("exec_sql", {"query": sql}).execute
                )
                logger.info(f"Executed: {sql[:50]}...")
            except Exception as e:
                # If exec_sql doesn't exist (production), just log
                logger.info(f"Table may already exist or use migrations: {sql[:50]}...")
    
    async def close(self) -> None:
        """Close database connections."""
        # Supabase client doesn't need explicit closing
        self._client = None


# Singleton instance
db_manager = DatabaseManager()