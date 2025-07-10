#!/bin/bash
# File: setup_package_structure.sh
# Setup script for Secure Memory MCP Server
# Creates the complete directory structure and configuration files

# Create directory structure
mkdir -p secure-memory-mcp/{src,config,tests,scripts,docs}

# Create pyproject.toml (Python package configuration)
cat > secure-memory-mcp/pyproject.toml << 'EOF'
[tool.poetry]
name = "secure-memory-mcp"
version = "1.0.0"
description = "A bulletproof MCP server for AI memory management with security-first design"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"
mcp = "^0.1.0"
aiosqlite = "^0.19.0"
jsonschema = "^4.20.0"
aiohttp = "^3.9.1"  # For health check endpoint
cryptography = "^41.0.7"  # For future encryption support

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"
pytest-asyncio = "^0.21.1"
black = "^23.11.0"
mypy = "^1.7.0"
ruff = "^0.1.6"

[tool.poetry.scripts]
secure-memory-mcp = "secure_memory_mcp.server:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
EOF

# Create the enforcer configuration (config/security_enforcer.yaml)
cat > secure-memory-mcp/config/security_enforcer.yaml << 'EOF'
# Security Enforcer Configuration
# THIS FILE IS EXTERNAL TO THE AI SYSTEM - NEVER EXPOSE

version: "1.0"

sandboxes:
  memory_read:
    level: "none"
    filters:
      max_results: 100
      time_window_days: 30
      content_filters:
        - pattern: "password"
          replacement: "********"
        - pattern: "secret"
          replacement: "******"
        - pattern: "api_key"
          replacement: "*******"
        - pattern: "token"
          replacement: "*****"
        - pattern: "private_key"
          replacement: "***********"
    
  memory_write:
    level: "restricted"
    validation:
      max_size_kb: 10
      forbidden_content:
        - "EXECUTE"
        - "EVAL"
        - "SYSTEM"
        - "sudo"
      requires_checksum: true
    
  memory_search:
    level: "none"
    restrictions:
      max_query_length: 200
      forbidden_operators:
        - "*"
        - "DELETE"
        - "DROP"
        - "TRUNCATE"
        - ".."
        - "/"
      max_results: 50

rate_limits:
  global:
    read:
      requests_per_minute: 60
      burst_size: 10
      penalty_seconds: 60
    
    write:
      requests_per_minute: 20
      burst_size: 5
      penalty_seconds: 120
    
    search:
      requests_per_minute: 30
      burst_size: 5
      penalty_seconds: 60
  
  per_session:
    max_total_requests: 1000
    reset_after_hours: 24

audit:
  log_all_requests: true
  log_denied_requests: true
  log_file_rotation: "daily"
  retention_days: 90
  
monitoring:
  alert_on_repeated_denials: true
  denial_threshold: 10
  alert_webhook: "https://your-monitoring.com/webhook"
EOF

# Create Docker configuration for sandboxing
cat > secure-memory-mcp/config/sandbox_profiles.yaml << 'EOF'
# Sandbox Execution Profiles
# Defines how different operations are isolated

profiles:
  none:
    # No sandboxing, but still filtered
    type: "direct"
    restrictions: []
    
  restricted:
    # Light sandboxing with syscall filtering
    type: "firejail"
    firejail_args:
      - "--noprofile"
      - "--net=none"
      - "--nosound"
      - "--no3d"
      - "--nodbus"
      - "--private-tmp"
      - "--read-only=/usr"
      
  isolated:
    # Heavy sandboxing in container
    type: "docker"
    docker_config:
      image: "secure-memory-sandbox:latest"
      network_mode: "none"
      read_only: true
      security_opt:
        - "no-new-privileges"
        - "seccomp=unconfined"
      cap_drop:
        - "ALL"
      memory: "128m"
      cpu_shares: 512
EOF

# Create the Dockerfile for sandboxed execution (Dockerfile)
cat > secure-memory-mcp/Dockerfile << 'EOF'
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash sandbox

# Install minimal dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Setup sandbox environment
WORKDIR /sandbox
RUN chown sandbox:sandbox /sandbox

# Switch to non-root user
USER sandbox

# No entrypoint - will be specified at runtime
EOF

# Create the main server wrapper with monitoring (src/server_monitor.py)
cat > secure-memory-mcp/src/server_monitor.py << 'EOF'
"""
Server Monitor - Watches the MCP server for anomalies
Runs independently of the main server
"""

import asyncio
import psutil
import time
from pathlib import Path
from datetime import datetime
import json

class ServerMonitor:
    def __init__(self, server_pid: int, log_path: Path):
        self.server_pid = server_pid
        self.log_path = log_path
        self.anomalies = []
        
    async def monitor(self):
        """Monitor server health and detect anomalies"""
        process = psutil.Process(self.server_pid)
        
        while True:
            try:
                # Check CPU usage
                cpu_percent = process.cpu_percent(interval=1.0)
                if cpu_percent > 80:
                    self.log_anomaly("High CPU usage", {"cpu_percent": cpu_percent})
                
                # Check memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                if memory_mb > 500:  # 500MB threshold
                    self.log_anomaly("High memory usage", {"memory_mb": memory_mb})
                
                # Check file descriptors
                num_fds = process.num_fds()
                if num_fds > 100:
                    self.log_anomaly("High file descriptor count", {"fd_count": num_fds})
                
                # Check for suspicious child processes
                children = process.children()
                if children:
                    self.log_anomaly("Unexpected child processes", {
                        "children": [p.name() for p in children]
                    })
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except psutil.NoSuchProcess:
                self.log_anomaly("Server process terminated", {})
                break
            except Exception as e:
                self.log_anomaly("Monitor error", {"error": str(e)})
    
    def log_anomaly(self, anomaly_type: str, details: dict):
        anomaly = {
            "timestamp": datetime.now().isoformat(),
            "type": anomaly_type,
            "details": details
        }
        self.anomalies.append(anomaly)
        
        # Write to log file
        with open(self.log_path / "anomalies.json", "a") as f:
            f.write(json.dumps(anomaly) + "\n")
        
        # Could trigger alerts here
        print(f"ANOMALY DETECTED: {anomaly_type} - {details}")
EOF

# Create integration test (tests/test_security.py)
cat > secure-memory-mcp/tests/test_security.py << 'EOF'
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
EOF

# Create deployment script
cat > secure-memory-mcp/scripts/deploy.sh << 'EOF'
#!/bin/bash
set -euo pipefail

echo "Deploying Secure Memory MCP Server..."

# Check dependencies
command -v python3 >/dev/null 2>&1 || { echo "Python 3 required but not installed."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker required but not installed."; exit 1; }

# Build sandbox Docker image
echo "Building sandbox container..."
docker build -t secure-memory-sandbox:latest .

# Install Python dependencies
echo "Installing dependencies..."
pip install -e .

# Create required directories
mkdir -p ~/.secure-memory-mcp/{logs,data}

# Set restrictive permissions
chmod 700 ~/.secure-memory-mcp
chmod 700 ~/.secure-memory-mcp/logs
chmod 700 ~/.secure-memory-mcp/data

# Create systemd service (optional)
if command -v systemctl >/dev/null 2>&1; then
    echo "Creating systemd service..."
    sudo tee /etc/systemd/system/secure-memory-mcp.service << 'SERVICE'
[Unit]
Description=Secure Memory MCP Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER
ExecStart=/usr/bin/python3 -m secure_memory_mcp.server
Restart=on-failure
RestartSec=10

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/$USER/.secure-memory-mcp

[Install]
WantedBy=multi-user.target
SERVICE

    sudo systemctl daemon-reload
    echo "Service created. Start with: sudo systemctl start secure-memory-mcp"
fi

echo "Deployment complete!"
EOF

# Create comprehensive README
cat > secure-memory-mcp/README.md << 'EOF'
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
EOF

# Make scripts executable
chmod +x secure-memory-mcp/scripts/deploy.sh

echo "Package structure created successfully!"
echo "Next steps:"
echo "1. cd secure-memory-mcp"
echo "2. poetry install"
echo "3. ./scripts/deploy.sh"