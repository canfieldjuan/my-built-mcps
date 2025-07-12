import json
import asyncio
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
import redis.asyncio as redis
from supabase import create_client, Client
from anthropic import AsyncAnthropic
import hashlib
import pickle
from enum import Enum
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import logging
from concurrent.futures import ThreadPoolExecutor
import uvicorn

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for sync operations
executor = ThreadPoolExecutor(max_workers=10)

class MemoryTier(str, Enum):
    SHORT_TERM = "short_term"
    MID_TERM = "mid_term"
    LONG_TERM = "long_term"
    META = "meta"

# Pydantic models for API
class AddMessageRequest(BaseModel):
    session_id: str
    role: str
    content: str
    tags: List[str] = Field(default_factory=list)

class GetContextRequest(BaseModel):
    session_id: str
    query: str
    max_tokens: int = 8192

class AutoRecallRequest(BaseModel):
    session_id: str
    query: str

class CreateMetaSummaryRequest(BaseModel):
    session_id: str
    period: str = Field(default="daily", pattern="^(daily|weekly|monthly)$")

class MCPToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class MCPRequest(BaseModel):
    method: str
    params: Optional[Dict[str, Any]] = None

# Fixed ProductionContextManager
class ProductionContextManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        
        # Initialize clients
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.redis_client: Optional[redis.Redis] = None
        self.anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        
        # ML models
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Configuration
        self.short_term_limit = 4096
        self.summarize_threshold = 2048
        self.cache_ttl = 3600
        
        # In-memory buffers
        self.short_term_buffer: List[Dict[str, Any]] = []
        self.short_term_tokens = 0
        
    async def initialize(self):
        """Initialize database tables and connections"""
        # Connect to Redis
        self.redis_client = await redis.from_url(REDIS_URL)
        
        # Create tables if needed
        await self._create_tables()
        
        # Create or update session
        await self._init_session()
    
    async def _create_tables(self):
        """Execute table creation via asyncio.to_thread"""
        sql_commands = [
            """
            CREATE TABLE IF NOT EXISTS context_nodes (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                tier TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                embedding vector(384),
                tokens INTEGER DEFAULT 0,
                tags TEXT[],
                parent_ids TEXT[],
                child_ids TEXT[],
                topic_anchors TEXT[],
                metadata JSONB,
                code_diff JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_nodes_session ON context_nodes(session_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_timestamp ON context_nodes(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_nodes_tier ON context_nodes(tier);
            CREATE INDEX IF NOT EXISTS idx_nodes_tags ON context_nodes USING GIN(tags);
            """,
            """
            CREATE TABLE IF NOT EXISTS topic_anchors (
                id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
                session_id TEXT NOT NULL,
                name TEXT NOT NULL,
                node_ids TEXT[],
                keywords TEXT[],
                embedding vector(384),
                last_accessed TIMESTAMPTZ DEFAULT NOW(),
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(session_id, name)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                id TEXT PRIMARY KEY,
                started_at TIMESTAMPTZ DEFAULT NOW(),
                last_activity TIMESTAMPTZ DEFAULT NOW(),
                metadata JSONB,
                total_tokens INTEGER DEFAULT 0,
                active BOOLEAN DEFAULT true
            );
            """
        ]
        
        # Execute table creation for dev/local setup
        try:
            for sql in sql_commands:
                try:
                    # For dev environments - execute the SQL
                    await asyncio.to_thread(
                        self.supabase.rpc("exec_sql", {"query": sql}).execute
                    )
                    logger.info(f"Executed table creation: {sql[:50]}...")
                except Exception as e:
                    # If exec_sql doesn't exist, just log
                    logger.info(f"Table may already exist or use migrations: {sql[:50]}...")
        except Exception as e:
            logger.warning(f"Table creation skipped (use migrations in production): {e}")
    
    async def _init_session(self):
        """Initialize or update session"""
        try:
            await asyncio.to_thread(
                self.supabase.table('conversation_sessions').upsert({
                    'id': self.session_id,
                    'last_activity': datetime.now().isoformat(),
                    'active': True
                }).execute
            )
        except Exception as e:
            logger.error(f"Session init error: {e}")
    
    def _get_cache_key(self, key: str) -> str:
        """Get session-scoped cache key"""
        return f"{self.session_id}:{key}"
    
    async def _get_cached_or_fetch(self, key: str, fetch_func, ttl: int = None):
        """Get from Redis cache or fetch from database"""
        if not self.redis_client:
            return await fetch_func()
        
        ttl = ttl or self.cache_ttl
        cache_key = self._get_cache_key(key)
        
        try:
            cached = await self.redis_client.get(cache_key)
            if cached:
                return pickle.loads(cached)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        result = await fetch_func()
        
        if result is not None:
            try:
                await self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    pickle.dumps(result)
                )
            except Exception as e:
                logger.warning(f"Cache set error: {e}")
        
        return result
    
    async def _invalidate_cache(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        if not self.redis_client:
            return
        
        try:
            pattern = self._get_cache_key(pattern)
            async for key in self.redis_client.scan_iter(match=pattern):
                await self.redis_client.delete(key)
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
    
    def _normalize_embedding(self, embedding: np.ndarray) -> List[float]:
        """Normalize embedding for pgvector"""
        # Ensure float32 and normalize
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()
    
    async def add_message(self, role: str, content: str, tags: List[str] = None):
        """Add a message to the context"""
        tokens = len(self.tokenizer.encode(content))
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or [],
            "tokens": tokens
        }
        
        self.short_term_buffer.append(message)
        self.short_term_tokens += tokens
        
        # Cache the buffer with session scope
        if self.redis_client:
            cache_key = self._get_cache_key("short_term_buffer")
            await self.redis_client.setex(
                cache_key,
                300,  # 5 minute TTL
                pickle.dumps(self.short_term_buffer)
            )
        
        # Check if we need to create a summary node
        if self.short_term_tokens >= self.summarize_threshold:
            await self.create_summary_node()
    
    async def create_summary_node(self):
        """Create a summary node and persist to Supabase"""
        if not self.short_term_buffer:
            return
        
        combined_content = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.short_term_buffer
        ])
        
        # Generate summary and topics
        summary = await self._generate_summary_with_llm(combined_content)
        topics = await self._extract_topics_with_llm(combined_content)
        
        # Create embedding with normalization
        embedding = self.encoder.encode([summary])[0]
        embedding = self._normalize_embedding(embedding)
        
        node_id = hashlib.md5(
            f"{self.session_id}:{combined_content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Create node dict
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
        
        # Save to Supabase with async wrapper
        await self._save_node_to_supabase(node_dict)
        
        # Update topic anchors
        for topic in topics:
            await self._update_topic_anchor(topic, node_id, summary)
        
        # Clear buffers
        self.short_term_buffer = []
        self.short_term_tokens = 0
        
        # Clean up Redis
        if self.redis_client:
            await self.redis_client.delete(self._get_cache_key("short_term_buffer"))
        
        # Invalidate relevant caches
        await self._invalidate_cache("context_window:*")
    
    async def _save_node_to_supabase(self, node_dict: Dict[str, Any]):
        """Save node to Supabase with proper async handling"""
        try:
            await asyncio.to_thread(
                self.supabase.table('context_nodes').insert(node_dict).execute
            )
            
            # Manage memory tiers
            await self._manage_memory_tiers()
            
        except Exception as e:
            logger.error(f"Error saving node: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def _update_topic_anchor(self, topic: str, node_id: str, content: str):
        """Update topic anchor with session scope"""
        try:
            # Check if topic exists for this session
            response = await asyncio.to_thread(
                self.supabase.table('topic_anchors')
                .select("*")
                .eq('session_id', self.session_id)
                .eq('name', topic)
                .execute
            )
            
            if response.data:
                # Update existing
                anchor = response.data[0]
                node_ids = anchor.get('node_ids', [])
                if node_id not in node_ids:
                    node_ids.append(node_id)
                
                # Update keywords safely
                words = content.lower().split()
                existing_keywords = anchor.get('keywords', [])
                new_keywords = [w for w in words if len(w) > 4 and w not in existing_keywords]
                keywords = (existing_keywords + new_keywords)[:50]
                
                # Safe access count increment
                current_count = anchor.get('access_count', 0)
                
                await asyncio.to_thread(
                    self.supabase.table('topic_anchors').update({
                        'node_ids': node_ids,
                        'keywords': keywords,
                        'last_accessed': datetime.now().isoformat(),
                        'access_count': current_count + 1
                    }).eq('id', anchor['id']).execute
                )
            else:
                # Create new
                words = content.lower().split()
                keywords = [w for w in words if len(w) > 4][:20]
                embedding = self._normalize_embedding(self.encoder.encode([topic])[0])
                
                await asyncio.to_thread(
                    self.supabase.table('topic_anchors').insert({
                        'session_id': self.session_id,
                        'name': topic,
                        'node_ids': [node_id],
                        'keywords': keywords,
                        'embedding': embedding
                    }).execute
                )
            
            # Invalidate topic cache
            await self._invalidate_cache(f"topic:{topic}")
            
        except Exception as e:
            logger.error(f"Topic anchor update error: {e}")
    
    async def _call_claude_with_retry(self, messages: List[Dict[str, str]], 
                                     max_tokens: int = 500, 
                                     temperature: float = 0.3,
                                     retries: int = 2) -> Optional[str]:
        """Call Claude API with retry logic"""
        for attempt in range(retries):
            try:
                response = await self.anthropic.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages
                )
                return response.content[0].text
            except Exception as e:
                logger.warning(f"Claude API attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    raise
        return None
    
    async def _generate_summary_with_llm(self, content: str) -> str:
        """Generate intelligent summary using Claude"""
        try:
            text = await self._call_claude_with_retry([{
                "role": "user",
                "content": f"""Summarize this conversation chunk concisely while preserving:
                1. Key technical decisions and implementations
                2. Important questions asked
                3. Code or architecture discussed
                4. Any tags or topics mentioned
                
                Conversation:
                {content[:4000]}
                
                Provide a 2-3 sentence summary."""
            }])
            return text if text else self._fallback_summary(content)
        except Exception as e:
            logger.error(f"LLM summary error after retries: {e}")
            return self._fallback_summary(content)
    
    def _fallback_summary(self, content: str) -> str:
        """Fallback summary generation"""
        lines = content.split('\n')
        return f"Summary of {len(lines)} messages discussing: {content[:200]}..."
    
    async def _extract_topics_with_llm(self, content: str) -> List[str]:
        """Extract topic anchors using LLM"""
        try:
            text = await self._call_claude_with_retry([{
                "role": "user",
                "content": f"""Extract 3-5 key topic anchors from this conversation.
                Return only lowercase topic names separated by commas, like: api_design, error_handling, performance
                
                Conversation:
                {content[:2000]}
                
                Topics:"""
            }], max_tokens=100, temperature=0.2)
            
            if not text:
                return self._fallback_topic_extraction(content)
            
            topics_str = text.strip()
            # Validate Claude response
            if not topics_str or ',' not in topics_str:
                return []
            return [f"topic_{topic.strip()}" for topic in topics_str.split(',') if topic.strip()][:5]
        except Exception as e:
            logger.error(f"LLM topic extraction error after retries: {e}")
            return self._fallback_topic_extraction(content)
    
    def _fallback_topic_extraction(self, content: str) -> List[str]:
        """Fallback topic extraction"""
        import re
        topics = set()
        tag_pattern = r'#(\w+)'
        topics.update(f"topic_{tag}" for tag in re.findall(tag_pattern, content.lower()))
        return list(topics)[:5]
    
    async def get_context_window(self, query: str, max_tokens: int = 8192) -> str:
        """Build optimized context window for session"""
        cache_key = f"context_window:{hashlib.md5(query.encode()).hexdigest()}:{max_tokens}"
        
        async def build_context():
            context_parts = []
            current_tokens = 0
            
            # 1. Get recent messages from cache/buffer
            if self.redis_client:
                try:
                    cached_buffer = await self.redis_client.get(
                        self._get_cache_key("short_term_buffer")
                    )
                    if cached_buffer:
                        buffer = pickle.loads(cached_buffer)
                        for msg in buffer[-10:]:
                            part = f"{msg['role']}: {msg['content']}"
                            tokens = msg.get('tokens', len(self.tokenizer.encode(part)))
                            if current_tokens + tokens < max_tokens * 0.3:
                                context_parts.append(part)
                                current_tokens += tokens
                except Exception as e:
                    logger.warning(f"Buffer cache error: {e}")
            
            # 2. Get recent mid-term summaries for this session
            try:
                recent_response = await asyncio.to_thread(
                    self.supabase.table('context_nodes')
                    .select("summary, content, tokens, timestamp")
                    .eq('session_id', self.session_id)
                    .eq('tier', MemoryTier.MID_TERM.value)
                    .order('timestamp', desc=True)
                    .limit(5)
                    .execute
                )
                
                for node in recent_response.data:
                    summary = node['summary'] or node['content'][:200]
                    tokens = node['tokens']
                    if current_tokens + tokens < max_tokens * 0.6:
                        timestamp = datetime.fromisoformat(node['timestamp'])
                        context_parts.append(
                            f"[Summary from {timestamp.strftime('%Y-%m-%d %H:%M')}]: {summary}"
                        )
                        current_tokens += tokens
            except Exception as e:
                logger.error(f"Mid-term fetch error: {e}")
            
            # 3. Get relevant context based on query
            relevant_nodes = await self.get_relevant_context(query, k=3)
            for node in relevant_nodes:
                if node.get('tier') == MemoryTier.LONG_TERM.value:
                    summary = node.get('summary') or node.get('content', '')[:100]
                    tokens = node.get('tokens', 0)
                    if current_tokens + tokens < max_tokens * 0.8:
                        context_parts.append(f"[Historical context]: {summary}")
                        current_tokens += tokens
            
            return "\n\n".join(context_parts)
        
        return await self._get_cached_or_fetch(cache_key, build_context, ttl=300)
    
    async def get_relevant_context(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Get relevant nodes using vector similarity search"""
        cache_key = f"relevant_context:{hashlib.md5(query.encode()).hexdigest()}:{k}"
        
        async def fetch():
            # Generate query embedding
            embedding = self._normalize_embedding(self.encoder.encode([query])[0])
            
            # Use raw SQL for vector search with session scope
            # This assumes you've created the appropriate function in Supabase
            try:
                response = await asyncio.to_thread(
                    self.supabase.rpc('match_session_context_nodes', {
                        'session_id': self.session_id,
                        'query_embedding': embedding,
                        'match_count': k,
                        'match_threshold': 0.7
                    }).execute
                )
                return response.data if response.data else []
            except Exception as e:
                logger.error(f"Vector search error: {e}")
                return []
        
        return await self._get_cached_or_fetch(cache_key, fetch, ttl=600)
    
    async def _manage_memory_tiers(self):
        """Promote old mid-term nodes to long-term for this session"""
        try:
            response = await asyncio.to_thread(
                self.supabase.table('context_nodes')
                .select("id, timestamp")
                .eq('session_id', self.session_id)
                .eq('tier', MemoryTier.MID_TERM.value)
                .order('timestamp')
                .execute
            )
            
            mid_term_nodes = response.data
            
            if len(mid_term_nodes) > 10:
                nodes_to_promote = mid_term_nodes[:-10]
                
                for node in nodes_to_promote:
                    await asyncio.to_thread(
                        self.supabase.table('context_nodes').update({
                            'tier': MemoryTier.LONG_TERM.value,
                            'updated_at': datetime.now().isoformat()
                        }).eq('id', node['id']).execute
                    )
                    
                    await self._invalidate_cache(f"node:{node['id']}")
                    
        except Exception as e:
            logger.error(f"Memory tier management error: {e}")

# Create the updated Supabase function for session-scoped search
SUPABASE_SESSION_FUNCTION = """
CREATE OR REPLACE FUNCTION match_session_context_nodes(
    session_id text,
    query_embedding vector(384),
    match_count int DEFAULT 10,
    match_threshold float DEFAULT 0.7
)
RETURNS TABLE (
    id text,
    content text,
    summary text,
    tier text,
    timestamp timestamptz,
    tokens int,
    tags text[],
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cn.id,
        cn.content,
        cn.summary,
        cn.tier,
        cn.timestamp,
        cn.tokens,
        cn.tags,
        1 - (cn.embedding <=> query_embedding) as similarity
    FROM context_nodes cn
    WHERE cn.session_id = session_id
      AND 1 - (cn.embedding <=> query_embedding) > match_threshold
    ORDER BY cn.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""

# Session manager with cleanup
class SessionManager:
    def __init__(self, max_sessions: int = 1000, session_ttl_hours: int = 24):
        self.sessions: Dict[str, ProductionContextManager] = {}
        self.session_access: Dict[str, datetime] = {}
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(hours=session_ttl_hours)
    
    async def get_or_create_session(self, session_id: str) -> ProductionContextManager:
        # Update access time
        self.session_access[session_id] = datetime.now()
        
        # Check if we need to clean up old sessions
        if len(self.sessions) >= self.max_sessions:
            await self._cleanup_old_sessions()
        
        if session_id not in self.sessions:
            manager = ProductionContextManager(session_id)
            await manager.initialize()
            self.sessions[session_id] = manager
            
        return self.sessions[session_id]
    
    async def _cleanup_old_sessions(self):
        """Remove inactive sessions"""
        now = datetime.now()
        sessions_to_remove = []
        
        for sid, last_access in self.session_access.items():
            if now - last_access > self.session_ttl:
                sessions_to_remove.append(sid)
        
        for sid in sessions_to_remove:
            if sid in self.sessions:
                # Clean up Redis keys for this session
                manager = self.sessions[sid]
                if manager.redis_client:
                    try:
                        await manager._invalidate_cache("*")
                    except:
                        pass
                
                del self.sessions[sid]
                del self.session_access[sid]
                logger.info(f"Cleaned up inactive session: {sid}")
    
    async def periodic_cleanup(self):
        """Run periodic cleanup task"""
        while True:
            await asyncio.sleep(3600)  # Run every hour
            await self._cleanup_old_sessions()

# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.session_manager = SessionManager()
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(app.state.session_manager.periodic_cleanup())
    
    logger.info("MCP Context Server started")
    yield
    # Shutdown
    cleanup_task.cancel()
    logger.info("MCP Context Server shutting down")

app = FastAPI(
    title="MCP Context Graph Server",
    version="2.0.0",
    lifespan=lifespan
)

# API Endpoints
@app.post("/api/messages")
async def add_message(request: AddMessageRequest):
    """Add a message to the context graph"""
    try:
        manager = await app.state.session_manager.get_or_create_session(request.session_id)
        await manager.add_message(request.role, request.content, request.tags)
        return {"success": True, "session_id": request.session_id}
    except Exception as e:
        logger.error(f"Add message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/context")
async def get_context(request: GetContextRequest):
    """Get optimized context window"""
    try:
        manager = await app.state.session_manager.get_or_create_session(request.session_id)
        context = await manager.get_context_window(request.query, request.max_tokens)
        tokens = len(manager.tokenizer.encode(context))
        return {
            "context": context,
            "tokens": tokens,
            "session_id": request.session_id
        }
    except Exception as e:
        logger.error(f"Get context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recall")
async def auto_recall(request: AutoRecallRequest):
    """Auto-recall relevant memories"""
    try:
        manager = await app.state.session_manager.get_or_create_session(request.session_id)
        # Implement auto_recall similar to before
        relevant_nodes = await manager.get_relevant_context(request.query, k=5)
        return {
            "relevant_nodes": relevant_nodes[:5],
            "session_id": request.session_id
        }
    except Exception as e:
        logger.error(f"Auto recall error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# MCP Protocol Endpoints
@app.post("/mcp")
async def handle_mcp_request(request: MCPRequest):
    """Handle MCP protocol requests"""
    method = request.method
    
    if method == "initialize":
        return {
            "server_info": {
                "name": "context-graph-server",
                "version": "2.0.0",
                "description": "Advanced MCP server with context graph"
            },
            "capabilities": {
                "tools": True,
                "context_graph": True,
                "memory_tiers": [tier.value for tier in MemoryTier],
                "sessions": True
            }
        }
    
    elif method == "list_tools":
        return {
            "tools": [
                {
                    "name": "add_message",
                    "description": "Add a message to the context graph",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string"},
                            "role": {"type": "string"},
                            "content": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["session_id", "role", "content"]
                    }
                },
                {
                    "name": "get_context",
                    "description": "Get optimized context window",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "session_id": {"type": "string"},
                            "query": {"type": "string"},
                            "max_tokens": {"type": "integer", "default": 8192}
                        },
                        "required": ["session_id", "query"]
                    }
                }
            ]
        }
    
    elif method == "call_tool":
        params = request.params or {}
        tool_call = MCPToolCall(**params)
        
        if tool_call.name == "add_message":
            req = AddMessageRequest(**tool_call.arguments)
            return await add_message(req)
        
        elif tool_call.name == "get_context":
            req = GetContextRequest(**tool_call.arguments)
            return await get_context(req)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_call.name}")
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

# WebSocket for real-time updates
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time updates
            await websocket.send_text(f"Session {session_id}: {data}")
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Basic test suite
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_add_message():
    """Test that buffer overflow creates a node"""
    manager = ProductionContextManager("test-session")
    await manager.initialize()
    
    # Add messages until we trigger summarization
    initial_buffer_size = len(manager.short_term_buffer)
    
    for i in range(10):
        await manager.add_message(
            "user", 
            "This is a test message with enough content to accumulate tokens. " * 50,
            ["test"]
        )
    
    # Buffer should be cleared after summarization
    assert len(manager.short_term_buffer) < initial_buffer_size + 10

@pytest.mark.asyncio
async def test_session_isolation():
    """Test that sessions are isolated"""
    manager1 = ProductionContextManager("session-1")
    manager2 = ProductionContextManager("session-2")
    
    await manager1.initialize()
    await manager2.initialize()
    
    await manager1.add_message("user", "Message in session 1", ["session1"])
    await manager2.add_message("user", "Message in session 2", ["session2"])
    
    # Get context for each session
    context1 = await manager1.get_context_window("test query")
    context2 = await manager2.get_context_window("test query")
    
    # Contexts should be different
    assert "session 1" in context1
    assert "session 2" in context2
    assert "session 1" not in context2
    assert "session 2" not in context1

@pytest.mark.asyncio
async def test_cache_invalidation():
    """Test cache invalidation logic"""
    manager = ProductionContextManager("test-cache")
    await manager.initialize()
    
    # Add to cache
    test_key = "test_key"
    test_data = {"test": "data"}
    
    async def fetch():
        return test_data
    
    # First call should fetch and cache
    result1 = await manager._get_cached_or_fetch(test_key, fetch, ttl=60)
    assert result1 == test_data
    
    # Second call should be from cache
    async def fetch_modified():
        return {"test": "modified"}
    
    result2 = await manager._get_cached_or_fetch(test_key, fetch_modified, ttl=60)
    assert result2 == test_data  # Still from cache
    
    # Invalidate cache
    await manager._invalidate_cache(test_key)
    
    # Now should fetch new data
    result3 = await manager._get_cached_or_fetch(test_key, fetch_modified, ttl=60)
    assert result3 == {"test": "modified"}

@pytest.mark.asyncio
async def test_api_endpoints():
    """Test FastAPI endpoints"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Test add message
        response = await client.post("/api/messages", json={
            "session_id": "test-api",
            "role": "user",
            "content": "Test message",
            "tags": ["test"]
        })
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Test get context
        response = await client.post("/api/context", json={
            "session_id": "test-api",
            "query": "test",
            "max_tokens": 1000
        })
        assert response.status_code == 200
        assert "context" in response.json()
        assert "tokens" in response.json()