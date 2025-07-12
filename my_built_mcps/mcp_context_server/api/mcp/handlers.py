"""
api/mcp/handlers.py

MCP protocol handlers.
"""
import logging
from fastapi import APIRouter, HTTPException

from ...models.schemas import MCPRequest, MCPToolCall, AddMessageRequest, GetContextRequest
from ...models.enums import MemoryTier
from ..v1.routes import add_message, get_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp")


@router.post("")
async def handle_mcp_request(request: MCPRequest):
    """Handle MCP protocol requests."""
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