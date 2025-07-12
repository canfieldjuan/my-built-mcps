"""
repositories/base.py

Base repository pattern implementation.
"""
import asyncio
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from .database import db_manager


class BaseRepository(ABC):
    """Base class for all repositories."""
    
    def __init__(self):
        self.db = db_manager.client
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """Return the table name for this repository."""
        pass
    
    async def _execute(self, query_func):
        """Execute a Supabase query with async wrapper."""
        return await asyncio.to_thread(query_func)
    
    async def insert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a record."""
        response = await self._execute(
            self.db.table(self.table_name).insert(data).execute
        )
        return response.data[0] if response.data else None
    
    async def update(self, id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a record by ID."""
        response = await self._execute(
            self.db.table(self.table_name).update(data).eq('id', id).execute
        )
        return response.data[0] if response.data else None
    
    async def delete(self, id: str) -> bool:
        """Delete a record by ID."""
        response = await self._execute(
            self.db.table(self.table_name).delete().eq('id', id).execute
        )
        return len(response.data) > 0 if response.data else False
    
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a record by ID."""
        response = await self._execute(
            self.db.table(self.table_name).select("*").eq('id', id).execute
        )
        return response.data[0] if response.data else None
    
    async def get_many(
        self,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        desc: bool = True
    ) -> List[Dict[str, Any]]:
        """Get multiple records with optional filtering and ordering."""
        query = self.db.table(self.table_name).select("*")
        
        # Apply filters
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        
        # Apply ordering
        if order_by:
            query = query.order(order_by, desc=desc)
        
        # Apply limit
        if limit:
            query = query.limit(limit)
        
        response = await self._execute(query.execute)
        return response.data if response.data else []