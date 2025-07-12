"""
repositories/topic_repository.py

Repository for topic anchor operations.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import BaseRepository


class TopicAnchorRepository(BaseRepository):
    """Repository for topic anchor CRUD operations."""
    
    @property
    def table_name(self) -> str:
        return "topic_anchors"
    
    async def get_or_create_topic(
        self,
        session_id: str,
        topic_name: str,
        embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Get existing topic or create new one."""
        # Check if topic exists
        response = await self._execute(
            self.db.table(self.table_name)
            .select("*")
            .eq('session_id', session_id)
            .eq('name', topic_name)
            .execute
        )
        
        if response.data:
            return response.data[0]
        
        # Create new topic
        new_topic = {
            'session_id': session_id,
            'name': topic_name,
            'node_ids': [],
            'keywords': [],
            'embedding': embedding,
            'access_count': 0
        }
        
        return await self.insert(new_topic)
    
    async def add_node_to_topic(
        self,
        session_id: str,
        topic_name: str,
        node_id: str,
        keywords: List[str] = None
    ) -> None:
        """Add a node to a topic anchor."""
        topic = await self.get_or_create_topic(session_id, topic_name)
        
        # Update node_ids
        node_ids = topic.get('node_ids', [])
        if node_id not in node_ids:
            node_ids.append(node_id)
        
        # Update keywords
        existing_keywords = topic.get('keywords', [])
        if keywords:
            new_keywords = [k for k in keywords if k not in existing_keywords]
            existing_keywords.extend(new_keywords[:50 - len(existing_keywords)])
        
        # Update topic
        await self.update(topic['id'], {
            'node_ids': node_ids,
            'keywords': existing_keywords,
            'last_accessed': datetime.now().isoformat(),
            'access_count': topic.get('access_count', 0) + 1
        })
    
    async def get_topics_by_session(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all topics for a session."""
        return await self.get_many(
            filters={'session_id': session_id},
            order_by='last_accessed',
            limit=limit,
            desc=True
        )
    
    async def get_nodes_for_topic(
        self,
        session_id: str,
        topic_name: str
    ) -> List[str]:
        """Get all node IDs associated with a topic."""
        response = await self._execute(
            self.db.table(self.table_name)
            .select("node_ids")
            .eq('session_id', session_id)
            .eq('name', topic_name)
            .execute
        )
        
        if response.data and response.data[0]:
            return response.data[0].get('node_ids', [])
        return []