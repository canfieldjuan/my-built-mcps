"""
core/context_manager.py

Production context manager - the main orchestrator for memory management.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..config.settings import settings
from ..models.enums import MemoryTier
from ..interfaces import (
    AbstractSummarizer, AbstractEmbedder, AbstractTokenizer,
    AbstractCache, AbstractContextStorage, AbstractVectorStore
)
from ..services.embedding_service import embedder
from ..services.llm_service import summarizer
from ..services.tokenizer_service import tokenizer
from ..cache.redis_manager import RedisCache, CacheManager
from ..repositories.context_repository import ContextNodeRepository
from ..repositories.topic_repository import TopicAnchorRepository
from ..repositories.database import db_manager
from ..policies.tier_policy import default_tier_policy
from ..policies.recall_policy import default_recall_policy
from ..utils.helpers import generate_node_id, extract_keywords_from_text, format_context_part
from .exceptions import ContextError

logger = logging.getLogger(__name__)


class ProductionContextManager:
    """Main context management orchestrator."""
    
    def __init__(
        self,
        session_id: str,
        summarizer_service: AbstractSummarizer = None,
        embedder_service: AbstractEmbedder = None,
        tokenizer_service: AbstractTokenizer = None,
        cache_service: AbstractCache = None,
        context_storage: AbstractContextStorage = None,
        vector_store: AbstractVectorStore = None
    ):
        self.session_id = session_id
        
        # Services (use defaults if not provided)
        self.summarizer = summarizer_service or summarizer
        self.embedder = embedder_service or embedder
        self.tokenizer = tokenizer_service or tokenizer
        
        # Storage
        self.context_repo = context_storage or ContextNodeRepository()
        self.topic_repo = TopicAnchorRepository()
        self.vector_store = vector_store or self.context_repo  # Same in our implementation
        
        # Cache
        self.cache = cache_service
        self.cache_manager = None
        
        # Configuration
        self.short_term_limit = settings.SHORT_TERM_LIMIT
        self.summarize_threshold = settings.SUMMARIZE_THRESHOLD
        
        # In-memory buffers
        self.short_term_buffer: List[Dict[str, Any]] = []
        self.short_term_tokens = 0
        
        # Policies
        self.tier_policy = default_tier_policy
        self.recall_policy = default_recall_policy
    
    async def initialize(self):
        """Initialize database tables and connections."""
        # Initialize database
        await db_manager.initialize_tables()
        
        # Initialize cache if not provided
        if not self.cache:
            self.cache = RedisCache()
            await self.cache.connect()
        
        self.cache_manager = CacheManager(self.cache)
        
        # Create or update session
        await self._init_session()
    
    async def _init_session(self):
        """Initialize or update session record."""
        try:
            await self.context_repo._execute(
                db_manager.client.table('conversation_sessions').upsert({
                    'id': self.session_id,
                    'last_activity': datetime.now().isoformat(),
                    'active': True
                }).execute
            )
        except Exception as e:
            logger.error(f"Session init error: {e}")
    
    async def add_message(self, role: str, content: str, tags: List[str] = None):
        """Add a message to the context."""
        tokens = self.tokenizer.count_tokens(content)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or [],
            "tokens": tokens
        }
        
        self.short_term_buffer.append(message)
        self.short_term_tokens += tokens
        
        # Cache the buffer
        await self.cache_manager.cache.set(
            self.cache_manager._get_cache_key(self.session_id, "short_term_buffer"),
            self.short_term_buffer,
            ttl=300  # 5 minute TTL
        )
        
        # Check if we need to create a summary node
        if self.short_term_tokens >= self.summarize_threshold:
            await self.create_summary_node()
    
    async def create_summary_node(self):
        """Create a summary node and persist to storage."""
        if not self.short_term_buffer:
            return
        
        combined_content = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.short_term_buffer
        ])
        
        # Generate summary and topics
        summary = await self.summarizer.summarize(combined_content)
        topics = await self.summarizer.extract_topics(combined_content)
        
        # Create embedding
        embedding_vector = self.embedder.encode([summary])[0]
        embedding = self.embedder.normalize_embedding(embedding_vector)
        
        node_id = generate_node_id(self.session_id, combined_content)
        
        # Create node
        node_dict = {
            'id': node_id,
            'session_id': self.session_id,
            'content': combined_content,
            'summary': summary,
            'tier': MemoryTier.MID_TERM.value,
            'timestamp': datetime.now().isoformat(),
            'embedding': embedding,
            'tokens': self.short_term_tokens,
            'tags': [tag for msg in self.short_term_buffer for tag in msg.get("tags", [])],
            'topic_anchors': topics
        }
        
        # Save to storage
        await self._save_node_to_storage(node_dict)
        
        # Update topic anchors
        for topic in topics:
            keywords = extract_keywords_from_text(summary)
            await self.topic_repo.add_node_to_topic(
                self.session_id,
                topic,
                node_id,
                keywords
            )
        
        # Clear buffers
        self.short_term_buffer = []
        self.short_term_tokens = 0
        
        # Clean up cache
        await self.cache_manager.cache.delete(
            self.cache_manager._get_cache_key(self.session_id, "short_term_buffer")
        )
        
        # Invalidate relevant caches
        await self.cache_manager.invalidate_session_cache(self.session_id, "context_window:*")
    
    async def _save_node_to_storage(self, node_dict: Dict[str, Any]):
        """Save node to storage with error handling."""
        try:
            await self.context_repo.save_node(node_dict)
            
            # Manage memory tiers
            await self._manage_memory_tiers()
            
        except Exception as e:
            logger.error(f"Error saving node: {e}")
            raise ContextError(f"Failed to save context node: {str(e)}")
    
    async def _manage_memory_tiers(self):
        """Promote old mid-term nodes to long-term."""
        try:
            promoted_count = await self.context_repo.promote_nodes_to_long_term(
                self.session_id,
                keep_recent=self.tier_policy.mid_term_retention_count
            )
            
            if promoted_count > 0:
                logger.info(f"Promoted {promoted_count} nodes to long-term for session {self.session_id}")
                # Invalidate caches for promoted nodes
                await self.cache_manager.invalidate_session_cache(self.session_id, "node:*")
                
        except Exception as e:
            logger.error(f"Memory tier management error: {e}")
    
    async def get_context_window(self, query: str, max_tokens: int = 8192) -> str:
        """Build optimized context window for session."""
        cache_key = f"context_window:{hashlib.md5(query.encode()).hexdigest()}:{max_tokens}"
        
        async def build_context():
            context_parts = []
            current_tokens = 0
            
            # Get token budgets
            budgets = self.recall_policy.allocate_token_budget(
                max_tokens,
                has_short_term=bool(self.short_term_buffer),
                has_relevant=True
            )
            tier_usage = {k: 0 for k in budgets.keys()}
            
            # 1. Get recent messages from buffer
            if self.short_term_buffer:
                for msg in self.short_term_buffer[-10:]:
                    part = f"{msg['role']}: {msg['content']}"
                    tokens = msg.get('tokens', self.tokenizer.count_tokens(part))
                    
                    if self.recall_policy.should_include_node(
                        {'tokens': tokens, 'tier': MemoryTier.SHORT_TERM.value},
                        current_tokens,
                        max_tokens,
                        budgets,
                        tier_usage
                    ):
                        context_parts.append(part)
                        current_tokens += tokens
                        tier_usage['short_term'] += tokens
            
            # 2. Get recent mid-term summaries
            recent_nodes = await self.context_repo.get_nodes_by_session(
                self.session_id,
                tier=MemoryTier.MID_TERM,
                limit=5
            )
            
            for node in recent_nodes:
                formatted = format_context_part(node)
                tokens = node['tokens']
                
                if self.recall_policy.should_include_node(
                    node,
                    current_tokens,
                    max_tokens,
                    budgets,
                    tier_usage
                ):
                    context_parts.append(formatted)
                    current_tokens += tokens
                    tier_usage['mid_term'] += tokens
            
            # 3. Get relevant context based on query
            relevant_nodes = await self.get_relevant_context(query, k=5)
            
            for node in relevant_nodes:
                if node.get('tier') == MemoryTier.LONG_TERM.value:
                    formatted = format_context_part(node)
                    tokens = node.get('tokens', 0)
                    
                    if self.recall_policy.should_include_node(
                        node,
                        current_tokens,
                        max_tokens,
                        budgets,
                        tier_usage
                    ):
                        context_parts.append(formatted)
                        current_tokens += tokens
                        tier_usage['long_term'] += tokens
            
            return "\n\n".join(context_parts)
        
        return await self.cache_manager.get_cached_or_fetch(
            self.session_id,
            cache_key,
            build_context,
            ttl=300
        )
    
    async def get_relevant_context(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Get relevant nodes using vector similarity search."""
        cache_key = f"relevant_context:{hashlib.md5(query.encode()).hexdigest()}:{k}"
        
        async def fetch():
            # Generate query embedding
            embedding_vector = self.embedder.encode([query])[0]
            embedding = self.embedder.normalize_embedding(embedding_vector)
            
            # Vector search
            try:
                results = await self.vector_store.search(
                    embedding=embedding,
                    k=k,
                    threshold=self.recall_policy.relevance_threshold,
                    filters={'session_id': self.session_id}
                )
                
                # Rank results
                return self.recall_policy.rank_nodes(results)
                
            except Exception as e:
                logger.error(f"Vector search error: {e}")
                return []
        
        return await self.cache_manager.get_cached_or_fetch(
            self.session_id,
            cache_key,
            fetch,
            ttl=600
        )
    
    async def cleanup(self):
        """Cleanup resources."""
        # Clear buffers
        self.short_term_buffer = []
        self.short_term_tokens = 0
        
        # Invalidate all session caches
        if self.cache_manager:
            await self.cache_manager.invalidate_session_cache(self.session_id, "*")


# Import hashlib at the top of the file
import hashlib