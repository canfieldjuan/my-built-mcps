"""
api/v1/websocket.py

WebSocket endpoint for real-time updates.
"""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...core.session_manager import session_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time session updates."""
    await websocket.accept()
    
    try:
        # Ensure session exists
        manager = await session_manager.get_or_create_session(session_id)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            # Echo back with session info
            response = {
                "type": "echo",
                "session_id": session_id,
                "message": data,
                "buffer_size": len(manager.short_term_buffer),
                "tokens": manager.short_term_tokens
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket.close()