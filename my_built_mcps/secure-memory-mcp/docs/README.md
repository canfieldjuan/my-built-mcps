# Secure Memory MCP Server

A bulletproof Model Context Protocol (MCP) server designed with security-first principles to prevent AI systems from controlling their own constraints.

## Key Security Features

### 1. **External Security Control**
- Security policies are loaded from external configuration files
- The AI system has no visibility into security rules
- All sandbox assignments are hardcoded and immutable

### 2. **Immutable Memory Store**
- Append-only design prevents tampering
- Each entry is checksummed for integrity verification
- Time-based access windows limit historical data exposure

### 3. **Multi-Layer Defense**
- Rate limiting on all operations
- Content filtering for sensitive information
- Query validation to prevent injection attacks
- Audit logging for all requests

### 4. **Sandboxing Support**
- Three levels: none (filtered), restricted (firejail), isolated (Docker)
- Sandbox level determined by operation type, not AI request
- Network isolation for high-risk operations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/secure-memory-mcp
cd secure-memory-mcp

# Run deployment script
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

## Configuration

Edit `config/security_enforcer.yaml` to adjust:
- Rate limits
- Content filters
- Time windows
- Sandbox assignments

**WARNING**: Never expose configuration files to the AI system.

## Usage

### Starting the Server
```bash
# Direct execution
python -m secure_memory_mcp.server

# Or via systemd
sudo systemctl start secure-memory-mcp
```

### MCP Client Configuration
```json
{
  "mcpServers": {
    "secure-memory": {
      "command": "python",
      "args": ["-m", "secure_memory_mcp.server"],
      "env": {}
    }
  }
}
```

### Available Tools

1. **memory_write** - Store information (rate-limited, size-limited)
2. **memory_read** - Retrieve information (filtered, time-windowed)
3. **memory_search** - Search content (query-validated, result-limited)

## Security Considerations

1. **Never** modify security configuration while the server is running
2. **Never** give the AI system access to configuration files
3. **Always** monitor audit logs for suspicious patterns
4. **Regularly** review and update content filters

## Monitoring

Check logs at:
- `~/.secure-memory-mcp/logs/audit.log` - All requests
- `~/.secure-memory-mcp/logs/anomalies.json` - Detected anomalies

## Architecture

```
External Config → Security Enforcer → Request
                         ↓
                  Rate Limiter
                         ↓
                  Content Filter
                         ↓
                  Sandbox Assignment
                         ↓
                  Memory Store (Immutable)
                         ↓
                  Filtered Response → AI System
```

## License

MIT License - See LICENSE file
