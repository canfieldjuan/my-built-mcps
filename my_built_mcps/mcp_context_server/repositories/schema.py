"""
repositories/schema.py

Database schema definitions for Supabase tables.
"""

# SQL table creation statements
CONTEXT_NODES_TABLE = """
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
"""

CONTEXT_NODES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_nodes_session ON context_nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_nodes_timestamp ON context_nodes(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_tier ON context_nodes(tier);
CREATE INDEX IF NOT EXISTS idx_nodes_tags ON context_nodes USING GIN(tags);
"""

TOPIC_ANCHORS_TABLE = """
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
"""

CONVERSATION_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id TEXT PRIMARY KEY,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    total_tokens INTEGER DEFAULT 0,
    active BOOLEAN DEFAULT true
);
"""

# Supabase function for vector similarity search
MATCH_SESSION_CONTEXT_NODES_FUNCTION = """
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

# All SQL commands in order
SQL_COMMANDS = [
    CONTEXT_NODES_TABLE,
    CONTEXT_NODES_INDEXES,
    TOPIC_ANCHORS_TABLE,
    CONVERSATION_SESSIONS_TABLE,
    MATCH_SESSION_CONTEXT_NODES_FUNCTION
]