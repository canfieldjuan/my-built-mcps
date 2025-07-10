# Secure Memory MCP - Hybrid Version Test Script
import asyncio
import json
import sys
from pathlib import Path

# Add the secure_memory directory to the path
sys.path.append(str(Path(__file__).parent / "secure_memory"))

from server_hybrid import SecureMemoryMCP

async def test_secure_memory():
    """Test the hybrid secure memory MCP server"""
    print("🔒 Testing Secure Memory MCP Server (Hybrid Version)")
    print("=" * 50)
    
    try:
        # Initialize server
        server = SecureMemoryMCP()
        await server.initialize()
        
        print("✅ Server initialized successfully")
        
        # Test memory write
        print("\n📝 Testing memory write...")
        write_result = await server.mcp.tools["memory_write"].func(
            "This is a test entry with sensitive data: password=secret123",
            {"category": "test", "priority": "high"}
        )
        print(f"Write result: {json.dumps(write_result, indent=2)}")
        
        # Test memory read
        print("\n📖 Testing memory read...")
        read_result = await server.mcp.tools["memory_read"].func()
        print(f"Read result: {json.dumps(read_result, indent=2)}")
        
        # Test memory search
        print("\n🔍 Testing memory search...")
        search_result = await server.mcp.tools["memory_search"].func("test entry")
        print(f"Search result: {json.dumps(search_result, indent=2)}")
        
        # Test memory status
        print("\n📊 Testing memory status...")
        status_result = await server.mcp.tools["memory_status"].func()
        print(f"Status result: {json.dumps(status_result, indent=2)}")
        
        # Test security violations
        print("\n🚨 Testing security violations...")
        
        # Test forbidden content
        violation_result = await server.mcp.tools["memory_write"].func(
            "exec('malicious code')",
            {"category": "malicious"}
        )
        print(f"Security violation result: {json.dumps(violation_result, indent=2)}")
        
        # Test rate limiting (multiple rapid requests)
        print("\n⏱️ Testing rate limiting...")
        for i in range(3):
            rate_result = await server.mcp.tools["memory_write"].func(
                f"Rate limit test {i}",
                {"test": "rate_limit"}
            )
            print(f"Rate limit test {i}: {rate_result.get('error', 'Success')}")
        
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
    asyncio.run(test_secure_memory())
