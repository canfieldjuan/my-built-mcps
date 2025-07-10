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
