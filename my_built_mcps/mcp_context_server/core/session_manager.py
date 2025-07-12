"""
core/session_manager.py

Session management with cleanup and lifecycle handling.
"""
import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from ..config.settings import settings
from .context_manager import ProductionContextManager
from .exceptions import SessionError

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages multiple concurrent sessions with lifecycle and cleanup."""
    
    def __init__(
        self,
        max_sessions: int = None,
        session_ttl_hours: int = None
    ):
        self.sessions: Dict[str, ProductionContextManager] = {}
        self.session_access: Dict[str, datetime] = {}
        self.max_sessions = max_sessions or settings.MAX_SESSIONS
        self.session_ttl = timedelta(hours=session_ttl_hours or settings.SESSION_TTL_HOURS)
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the session manager and cleanup task."""
        self._cleanup_task = asyncio.create_task(self.periodic_cleanup())
        logger.info("Session manager started")
    
    async def stop(self):
        """Stop the session manager and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup all sessions
        for session_id in list(self.sessions.keys()):
            await self._cleanup_session(session_id)
        
        logger.info("Session manager stopped")
    
    async def get_or_create_session(self, session_id: str) -> ProductionContextManager:
        """Get existing session or create a new one."""
        # Update access time
        self.session_access[session_id] = datetime.now()
        
        # Check if we need to clean up old sessions
        if len(self.sessions) >= self.max_sessions:
            await self._cleanup_old_sessions()
        
        # Return existing session if available
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Create new session
        try:
            manager = ProductionContextManager(session_id)
            await manager.initialize()
            self.sessions[session_id] = manager
            logger.info(f"Created new session: {session_id}")
            return manager
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            raise SessionError(f"Failed to create session: {str(e)}")
    
    async def get_session(self, session_id: str) -> Optional[ProductionContextManager]:
        """Get an existing session without creating."""
        if session_id in self.sessions:
            self.session_access[session_id] = datetime.now()
            return self.sessions[session_id]
        return None
    
    async def remove_session(self, session_id: str) -> bool:
        """Explicitly remove a session."""
        if session_id in self.sessions:
            await self._cleanup_session(session_id)
            return True
        return False
    
    async def _cleanup_session(self, session_id: str):
        """Clean up a single session."""
        if session_id not in self.sessions:
            return
        
        try:
            manager = self.sessions[session_id]
            await manager.cleanup()
            
            # Disconnect cache if it's session-specific
            if hasattr(manager.cache, 'disconnect'):
                await manager.cache.disconnect()
            
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
        finally:
            # Remove from tracking
            self.sessions.pop(session_id, None)
            self.session_access.pop(session_id, None)
            logger.info(f"Cleaned up session: {session_id}")
    
    async def _cleanup_old_sessions(self):
        """Remove inactive sessions based on TTL."""
        now = datetime.now()
        sessions_to_remove = []
        
        # Find expired sessions
        for sid, last_access in self.session_access.items():
            if now - last_access > self.session_ttl:
                sessions_to_remove.append(sid)
        
        # Remove expired sessions
        for sid in sessions_to_remove:
            await self._cleanup_session(sid)
        
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} inactive sessions")
    
    async def periodic_cleanup(self):
        """Run periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions."""
        now = datetime.now()
        
        stats = {
            'total_sessions': len(self.sessions),
            'max_sessions': self.max_sessions,
            'session_ttl_hours': self.session_ttl.total_seconds() / 3600,
            'sessions': []
        }
        
        for sid, last_access in self.session_access.items():
            age = now - last_access
            stats['sessions'].append({
                'id': sid,
                'last_access': last_access.isoformat(),
                'age_minutes': age.total_seconds() / 60,
                'active': sid in self.sessions
            })
        
        return stats


# Global session manager instance
session_manager = SessionManager()