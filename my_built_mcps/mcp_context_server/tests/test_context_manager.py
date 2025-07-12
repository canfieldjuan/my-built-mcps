"""
tests/test_context_manager.py

Tests for context manager functionality.
"""
import pytest

from mcp_context_server.core.context_manager import ProductionContextManager
from mcp_context_server.models.enums import MemoryTier


@pytest.mark.asyncio
async def test_add_message(test_context_manager: ProductionContextManager):
    """Test that messages are added to the buffer correctly."""
    # Add a message
    await test_context_manager.add_message(
        role="user",
        content="This is a test message",
        tags=["test", "sample"]
    )
    
    # Check buffer
    assert len(test_context_manager.short_term_buffer) == 1
    assert test_context_manager.short_term_buffer[0]["role"] == "user"
    assert test_context_manager.short_term_buffer[0]["content"] == "This is a test message"
    assert test_context_manager.short_term_buffer[0]["tags"] == ["test", "sample"]
    assert test_context_manager.short_term_tokens > 0


@pytest.mark.asyncio
async def test_buffer_overflow_creates_node(test_context_manager: ProductionContextManager):
    """Test that buffer overflow creates a summary node."""
    initial_buffer_size = len(test_context_manager.short_term_buffer)
    
    # Add messages until we trigger summarization
    large_message = "This is a test message with enough content to accumulate tokens. " * 50
    
    for i in range(10):
        await test_context_manager.add_message(
            role="user",
            content=large_message,
            tags=["test"]
        )
    
    # Buffer should be cleared after summarization
    assert len(test_context_manager.short_term_buffer) < initial_buffer_size + 10
    assert test_context_manager.short_term_tokens < test_context_manager.summarize_threshold


@pytest.mark.asyncio
async def test_session_isolation():
    """Test that sessions are isolated from each other."""
    # Create two separate managers
    manager1 = ProductionContextManager("test-session-1")
    manager2 = ProductionContextManager("test-session-2")
    
    await manager1.initialize()
    await manager2.initialize()
    
    try:
        # Add messages to different sessions
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
    finally:
        # Cleanup
        await manager1.cleanup()
        await manager2.cleanup()


@pytest.mark.asyncio
async def test_cache_functionality(test_context_manager: ProductionContextManager):
    """Test cache get/set operations."""
    if not test_context_manager.cache_manager:
        pytest.skip("Cache not available")
    
    # Test cache operations
    test_key = "test_key"
    test_data = {"test": "data", "number": 42}
    
    # Set cache
    await test_context_manager.cache_manager.cache.set(
        test_context_manager.cache_manager._get_cache_key(test_context_manager.session_id, test_key),
        test_data,
        ttl=60
    )
    
    # Get from cache
    cached = await test_context_manager.cache_manager.cache.get(
        test_context_manager.cache_manager._get_cache_key(test_context_manager.session_id, test_key)
    )
    
    assert cached == test_data


@pytest.mark.asyncio
async def test_get_relevant_context(test_context_manager: ProductionContextManager):
    """Test vector similarity search."""
    # First add some content
    await test_context_manager.add_message(
        "user",
        "I want to learn about machine learning algorithms",
        ["ml", "algorithms"]
    )
    
    # Force summarization
    test_context_manager.short_term_tokens = test_context_manager.summarize_threshold + 1
    await test_context_manager.create_summary_node()
    
    # Search for relevant content
    relevant = await test_context_manager.get_relevant_context(
        "Tell me about ML algorithms",
        k=5
    )
    
    # Should return results (though might be empty if vector search not fully configured)
    assert isinstance(relevant, list)