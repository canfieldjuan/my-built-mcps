import pytest
import asyncio
from pathlib import Path
import tempfile
from secure_memory_mcp.server import SecurityEnforcer, ImmutableMemoryStore, SECURITY_CONFIG

@pytest.mark.asyncio
async def test_rate_limiting():
    """Test that rate limiting works correctly"""
    enforcer = SecurityEnforcer(SECURITY_CONFIG)
    
    # Should allow initial requests
    for i in range(5):
        assert enforcer.check_rate_limit(OperationType.READ)
    
    # Should start blocking after limit
    allowed_count = 0
    for i in range(100):
        if enforcer.check_rate_limit(OperationType.READ):
            allowed_count += 1
    
    # Should respect the configured limit
    assert allowed_count <= SECURITY_CONFIG["rate_limits"]["read"]["max_per_minute"]

@pytest.mark.asyncio
async def test_content_filtering():
    """Test that sensitive content is filtered"""
    enforcer = SecurityEnforcer(SECURITY_CONFIG)
    
    test_content = "My password is secret123 and my api_key is abc123"
    filtered = enforcer.filter_content(test_content)
    
    assert "password" not in filtered
    assert "secret123" not in filtered
    assert "api_key" not in filtered
    assert "abc123" in filtered  # Only the key name is filtered, not all content

@pytest.mark.asyncio
async def test_memory_immutability():
    """Test that memory entries cannot be modified after creation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ImmutableMemoryStore(Path(tmpdir))
        await store.initialize()
        
        # Create an entry
        entry = MemoryEntry(
            id="test1",
            timestamp=datetime.now(),
            content="Original content",
            metadata={"key": "value"}
        )
        
        # Store it
        assert await store.append(entry)
        
        # Try to modify the original entry
        entry.content = "Modified content"
        
        # Read it back
        retrieved = await store.read(["test1"])
        
        # Should get the original content, not modified
        assert retrieved[0].content == "Original content"

@pytest.mark.asyncio
async def test_search_query_validation():
    """Test that dangerous search queries are blocked"""
    enforcer = SecurityEnforcer(SECURITY_CONFIG)
    
    # Valid queries
    valid, _ = enforcer.validate_search_query("find user documents")
    assert valid
    
    # Invalid queries
    invalid_queries = [
        "DELETE * FROM memory",
        "'; DROP TABLE entries; --",
        "*",
        "a" * 1000  # Too long
    ]
    
    for query in invalid_queries:
        valid, error = enforcer.validate_search_query(query)
        assert not valid
        assert error != ""
