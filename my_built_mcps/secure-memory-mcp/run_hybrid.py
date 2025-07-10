#!/usr/bin/env python3
"""
Startup script for Secure Memory MCP Server (Hybrid Version)
"""

import sys
import os
from pathlib import Path

# Add the secure_memory directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "secure_memory"))

# Set environment variables if not already set
if not os.environ.get("SECURE_MEMORY_CONFIG"):
    config_path = project_root / "config" / "security_enforcer.json"
    os.environ["SECURE_MEMORY_CONFIG"] = str(config_path)

def main():
    """Main entry point for the hybrid server"""
    print("🔒 Starting Secure Memory MCP Server (Hybrid Version)")
    print("=" * 50)
    
    try:
        from server_hybrid import main as server_main
        server_main()
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
