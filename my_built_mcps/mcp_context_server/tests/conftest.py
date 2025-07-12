"""
tests/conftest.py

Test configuration and fixtures.
"""
import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient

from mcp_context_server.main import app
from mcp_context_server.core.context_manager import ProductionContextManager
from mcp_context_server.core.session_manager import session_manager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """Create a test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
async def test_session_id() -> str:
    """Generate a unique test session ID."""
    import uuid
    return f"test-session-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def test_context_manager(test_session_id: str) -> AsyncGenerator[ProductionContextManager, None]:
    """Create a test context manager."""
    manager = ProductionContextManager(test_session_id)
    await manager.initialize()
    yield manager
    await manager.cleanup()


@pytest.fixture(autouse=True)
async def cleanup_sessions():
    """Clean up test sessions after each test."""
    yield
    # Clean up any test sessions
    for session_id in list(session_manager.sessions.keys()):
        if session_id.startswith("test-"):
            await session_manager.remove_session(session_id)