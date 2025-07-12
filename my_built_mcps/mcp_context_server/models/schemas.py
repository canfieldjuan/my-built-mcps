"""
models/schemas.py

Pydantic models for API requests and responses.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AddMessageRequest(BaseModel):
    """Request model for adding a message."""
    session_id: str
    role: str
    content: str
    tags: List[str] = Field(default_factory=list)


class GetContextRequest(BaseModel):
    """Request model for getting context."""
    session_id: str
    query: str
    max_tokens: int = 8192


class AutoRecallRequest(BaseModel):
    """Request model for auto-recall."""
    session_id: str
    query: str


class CreateMetaSummaryRequest(BaseModel):
    """Request model for creating meta summaries."""
    session_id: str
    period: str = Field(default="daily", pattern="^(daily|weekly|monthly)$")


class MCPToolCall(BaseModel):
    """Model for MCP tool calls."""
    name: str
    arguments: Dict[str, Any]


class MCPRequest(BaseModel):
    """Model for MCP protocol requests."""
    method: str
    params: Optional[Dict[str, Any]] = None


# Response models
class MessageResponse(BaseModel):
    """Response model for message operations."""
    success: bool
    session_id: str


class ContextResponse(BaseModel):
    """Response model for context retrieval."""
    context: str
    tokens: int
    session_id: str


class RecallResponse(BaseModel):
    """Response model for recall operations."""
    relevant_nodes: List[Dict[str, Any]]
    session_id: str


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    timestamp: str