# Secure Memory MCP - Hybrid Version 🔒

## Overview

This is a **hybrid FastMCP + Security** implementation of the Secure Memory MCP server. It combines the tamper-proof security architecture with the modern FastMCP framework for optimal performance and reliability.

## Key Features

### 🛡️ Security Architecture (Preserved)
- **Tamper-proof memory storage** - Prevents LLMs from rewriting security constraints
- **HMAC integrity verification** - Detects any tampering with stored data
- **Rate limiting** - Prevents abuse and DoS attacks
- **Content filtering** - Automatically redacts sensitive information
- **Audit logging** - Comprehensive security event tracking
- **Session management** - Secure session handling with timeouts

### 🚀 FastMCP Integration (New)
- **Modern MCP protocol** - Uses the latest FastMCP SDK
- **Simplified tool registration** - Clean decorator-based approach
- **Better error handling** - Improved error reporting and debugging
- **Context logging** - Integrated MCP context for better traceability
- **Async/await support** - Proper async handling throughout

## Installation

1. Install dependencies:
```bash
pip install -r requirements_hybrid.txt
```

2. Test the server:
```bash
python test_hybrid.py
```

## Configuration

The server uses the same security configuration as the original:
- Default config created at `~/.secure-memory-mcp/config/security_enforcer.json`
- Environment variable: `SECURE_MEMORY_CONFIG`
- Comprehensive rate limiting, content filtering, and audit settings

## Usage with Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "secure-memory": {
      "command": "python",
      "args": [
        "path/to/secure_memory/server_hybrid.py"
      ],
      "env": {
        "SECURE_MEMORY_CONFIG": "path/to/your/config.json"
      }
    }
  }
}
```

## Available Tools

### 🔐 memory_write
Store information securely with integrity protection.
- **Input**: `content` (string), `metadata` (optional dict)
- **Security**: Content validation, size limits, pattern filtering
- **Output**: Success status, ID, timestamp, checksum

### 📖 memory_read
Read stored information with automatic filtering.
- **Input**: `ids` (optional list of IDs)
- **Security**: Time window filtering, content redaction
- **Output**: Filtered entries with metadata

### 🔍 memory_search
Search memory with validated queries.
- **Input**: `query` (string)
- **Security**: Query validation, injection prevention
- **Output**: Matching entries with highlights

### 📊 memory_status
Get system status and statistics.
- **Input**: None
- **Output**: System health, entry count, configuration

## Security Features

### Rate Limiting
- **Read**: 60/minute, 10 burst
- **Write**: 20/minute, 5 burst  
- **Search**: 30/minute, 5 burst
- **Cooldown**: Automatic cooldown on violations

### Content Filtering
- **Automatic redaction** of passwords, API keys, secrets
- **Pattern detection** for malicious code
- **Size limits** and validation
- **Email address masking**

### Integrity Protection
- **SHA-256 checksums** for all entries
- **HMAC signatures** for tamper detection
- **Immutable storage** - entries cannot be modified
- **Audit trail** for all operations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Desktop                           │
└─────────────────────────┬───────────────────────────────────┘
                          │ MCP Protocol
┌─────────────────────────▼───────────────────────────────────┐
│                     FastMCP                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │               Security Enforcer                         ││
│  │  • Rate Limiting     • Content Filtering               ││
│  │  • Session Management • Input Validation               ││
│  │  • Audit Logging     • Pattern Detection               ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │            Immutable Memory Store                       ││
│  │  • SQLite Database   • HMAC Signatures                 ││
│  │  • Integrity Checks  • Tamper Detection                ││
│  │  • Audit Logs        • Checksums                       ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Improvements over Original

### ✅ Fixed Issues
- **Modern MCP SDK** - Uses current FastMCP instead of deprecated APIs
- **Simplified protocol handling** - FastMCP manages all protocol details
- **Better error handling** - Comprehensive error reporting
- **Cleaner code structure** - Separated concerns, better maintainability

### ✅ Enhanced Features
- **Context logging** - Better debugging and traceability
- **Async throughout** - Proper async/await patterns
- **Tool validation** - Automatic input validation via FastMCP
- **Better testing** - Included test script for verification

### ✅ Maintained Security
- **All security features preserved** - No reduction in security posture
- **Same configuration** - Compatible with existing setups
- **Tamper-proof design** - Core security architecture unchanged
- **Audit compliance** - All logging and monitoring retained

## Testing

Run the test script to verify functionality:

```bash
python test_hybrid.py
```

This will test:
- ✅ Server initialization
- ✅ Memory write operations
- ✅ Memory read with filtering
- ✅ Memory search functionality
- ✅ Security violation detection
- ✅ Rate limiting enforcement
- ✅ System status reporting

## Debugging

Use the MCP inspector for development:

```bash
mcp dev server_hybrid.py
```

This opens a web interface for testing tools directly.

## Why This Hybrid Approach?

1. **Preserves Security Intent** - The original security design is brilliant for preventing LLM manipulation
2. **Modernizes Implementation** - Uses current MCP best practices
3. **Improves Maintainability** - Cleaner code, better error handling
4. **Enhances Reliability** - Proven FastMCP framework reduces protocol bugs
5. **Maintains Compatibility** - Same configuration and behavior

This hybrid version gives you the tamper-proof security you designed while leveraging the modern MCP ecosystem for better reliability and maintainability.
