# Corrected test script for the fixed hybrid server
import asyncio
import json
import sys
from pathlib import Path

# Add the secure_memory directory to the path
sys.path.append(str(Path(__file__).parent / "secure_memory"))

async def test_secure_memory_fixed():
    """Test the corrected hybrid secure memory MCP server"""
    print("🔒 Testing Secure Memory MCP Server (Fixed Hybrid Version)")
    print("=" * 60)
    
    try:
        # Import and initialize server
        from server_hybrid_fixed import SecureMemoryMCP, memory_write, memory_read, memory_search, memory_status
        
        server = SecureMemoryMCP()
        await server.initialize()
        
        print("✅ Server initialized successfully")
        
        # Test memory write
        print("\n📝 Testing memory write...")
        write_result = await memory_write(
            "This is a test entry with sensitive data: password=secret123",
            {"category": "test", "priority": "high"}
        )
        print(f"Write result: {json.dumps(write_result, indent=2)}")
        
        # Test memory read
        print("\n📖 Testing memory read...")
        read_result = await memory_read()
        print(f"Read result: {json.dumps(read_result, indent=2)}")
        
        # Test memory search
        print("\n🔍 Testing memory search...")
        search_result = await memory_search("test entry")
        print(f"Search result: {json.dumps(search_result, indent=2)}")
        
        # Test memory status
        print("\n📊 Testing memory status...")
        status_result = await memory_status()
        print(f"Status result: {json.dumps(status_result, indent=2)}")
        
        # Test security violations
        print("\n🚨 Testing security violations...")
        
        # Test forbidden content
        violation_result = await memory_write(
            "exec('malicious code')",
            {"category": "malicious"}
        )
        print(f"Security violation result: {json.dumps(violation_result, indent=2)}")
        
        # Test oversized content
        large_content = "x" * (11 * 1024)  # 11KB (over 10KB limit)
        size_violation_result = await memory_write(large_content)
        print(f"Size violation result: {json.dumps(size_violation_result, indent=2)}")
        
        # Test forbidden search terms
        search_violation_result = await memory_search("'; DROP TABLE")
        print(f"Search violation result: {json.dumps(search_violation_result, indent=2)}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            await server.shutdown()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_secure_memory_fixed())
