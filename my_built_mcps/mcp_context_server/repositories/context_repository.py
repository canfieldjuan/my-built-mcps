"""
repositories/context_repository.py

Repository for context node operations.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseRepository
from ..interfaces import AbstractContextStorage, AbstractVectorStore
from ..models.enums import MemoryTier


class ContextNodeRepository(BaseRepository, AbstractContextStorage, AbstractVectorStore):
    """Repository for context node CRUD operations."""
    
    @property
    def table_name(self) -> str:
        return "context_nodes"
    
    async def save_node(self, node: Dict[str, Any]) -> str:
        """Save a context node and return its ID."""
        # Ensure timestamp is ISO format string
        if 'timestamp' not in node:
            node['timestamp'] = datetime.now().isoformat()
        
        result = await self.insert(node)
        return result['id'] if result else None
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a context node by ID."""
        return await self.get_by_id(node_id)
    
    async def update_node(self, node_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing node."""
        updates['updated_at'] = datetime.now().isoformat()
        await self.update(node_id, updates)
    
    async def query_nodes(
        self,
        filters: Dict[str, Any],
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query nodes with filters."""
        return await self.get_many(filters, order_by, limit)
    
    async def get_nodes_by_session(
        self,
        session_id: str,
        tier: Optional[MemoryTier] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get nodes for a specific session and optional tier."""
        filters = {'session_id': session_id}
        if tier:
            filters['tier'] = tier.value
        
        return await self.get_many(
            filters=filters,
            order_by='timestamp',
            limit=limit,
            desc=True
        )
    
    async def promote_nodes_to_long_term(
        self,
        session_id: str,
        keep_recent: int = 10
    ) -> int:
        """Promote old mid-term nodes to long-term."""
        # Get all mid-term nodes for session
        mid_term_nodes = await self.get_nodes_by_session(
            session_id,
            tier=MemoryTier.MID_TERM
        )
        
        if len(mid_term_nodes) <= keep_recent:
            return 0
        
        # Promote older nodes
        nodes_to_promote = mid_term_nodes[keep_recent:]
        promoted_count = 0
        
        for node in nodes_to_promote:
            await self.update_node(
                node['id'],
                {'tier': MemoryTier.LONG_TERM.value}
            )
            promoted_count += 1
        
        return promoted_count
    
    # Vector store operations
    async def search(
        self,
        embedding: List[float],
        k: int = 10,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using Supabase RPC."""
        # Default to session_id filter if provided
        session_id = filters.get('session_id') if filters else None
        
        if not session_id:
            raise ValueError("session_id is required for vector search")
        
        response = await self._execute(
            self.db.rpc('match_session_context_nodes', {
                'session_id': session_id,
                'query_embedding': embedding,
                'match_count': k,
                'match_threshold': threshold
            }).execute
        )
        
        return response.data if response.data else []
    
    async def upsert(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Insert or update a vector - handled by save_node in this implementation."""
        # In our implementation, embeddings are part of the node data
        metadata['embedding'] = embedding
        metadata['id'] = id
        await self.save_node(metadata)