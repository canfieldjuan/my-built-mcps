"""
api/v1/routes.py

API v1 routes for context management.
"""
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends

from ...models.schemas import (
    AddMessageRequest, GetContextRequest, AutoRecallRequest,
    MessageResponse, ContextResponse, RecallResponse, HealthResponse
)
from ...core.session_manager import session_manager
from ...services.tokenizer_service import tokenizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["context"])


@router.post("/messages", response_model=MessageResponse)
async def add_message(request: AddMessageRequest):
    """Add a message to the context graph."""
    try:
        manager = await session_manager.get_or_create_session(request.session_id)
        await manager.add_message(request.role, request.content, request.tags)
        return MessageResponse(success=True, session_id=request.session_id)
    except Exception as e:
        logger.error(f"Add message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/context", response_model=ContextResponse)
async def get_context(request: GetContextRequest):
    """Get optimized context window."""
    try:
        manager = await session_manager.get_or_create_session(request.session_id)
        context = await manager.get_context_window(request.query, request.max_tokens)
        tokens = tokenizer.count_tokens(context)
        return ContextResponse(
            context=context,
            tokens=tokens,
            session_id=request.session_id
        )
    except Exception as e:
        logger.error(f"Get context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recall", response_model=RecallResponse)
async def auto_recall(request: AutoRecallRequest):
    """Auto-recall relevant memories."""
    try:
        manager = await session_manager.get_or_create_session(request.session_id)
        relevant_nodes = await manager.get_relevant_context(request.query, k=5)
        return RecallResponse(
            relevant_nodes=relevant_nodes[:5],
            session_id=request.session_id
        )
    except Exception as e:
        logger.error(f"Auto recall error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session."""
    manager = await session_manager.get_session(session_id)
    if not manager:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "active": True,
        "short_term_messages": len(manager.short_term_buffer),
        "short_term_tokens": manager.short_term_tokens
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data."""
    removed = await session_manager.remove_session(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"success": True, "session_id": session_id}


@router.get("/sessions")
async def get_all_sessions():
    """Get statistics about all active sessions."""
    return session_manager.get_session_stats()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )