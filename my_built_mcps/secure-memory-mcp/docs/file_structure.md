# 📁 Secure Memory MCP - Complete File Structure

**File: docs/file_structure.md**  
**Purpose: Complete overview of all files and their locations**

## Project Directory Structure

```
secure-memory-mcp/
│
├── src/                                 # Source code directory
│   ├── secure_memory_mcp/
│   │   ├── __init__.py                 # Package initialization
│   │   └── server.py                   # Main MCP server implementation
│   └── server_monitor.py               # Optional monitoring wrapper
│
├── config/                             # Configuration files
│   ├── security_enforcer.yaml          # External security configuration
│   └── sandbox_profiles.yaml           # Docker/sandboxing profiles
│
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── test_security.py                # Comprehensive security tests
│   └── load_test.py                    # Performance testing
│
├── scripts/                            # Deployment and operations
│   ├── deploy.sh                       # Production deployment script
│   └── backup.sh                       # Automated backup script
│
├── docs/                               # Documentation
│   ├── README.md                       # Main project documentation
│   ├── production_checklist.md         # Launch checklist
│   ├── fixes_and_improvements.md       # Production fixes applied
│   └── file_structure.md               # This file
│
├── monitoring/                         # Monitoring tools
│   └── dashboard.html                  # Real-time monitoring UI
│
├── admin/                              # Admin tools (created by deploy)
│   └── status.sh                       # Service status checker
│
├── pyproject.toml                      # Python package configuration
├── Dockerfile                          # Container for sandboxing
├── claude_desktop_config.json          # MCP client configuration
└── .gitignore                          # Git ignore rules
```

## System Installation Paths

After running `deploy.sh`, files are installed to:

```
/opt/secure-memory-mcp/                 # Application directory
├── venv/                               # Python virtual environment
├── secure_memory_mcp/                  # Installed package
├── admin/                              # Admin scripts
│   └── status.sh
└── backup.sh                           # Backup script

/etc/secure-memory-mcp/                 # Configuration directory
└── security_enforcer.json              # Production security config

/var/lib/secure-memory-mcp/             # Data directory
├── memory.db                           # SQLite database
└── tmp/                                # Temporary files

/var/log/secure-memory-mcp/             # Log directory
├── secure-memory.log                   # Main application log
├── security.log                        # Security events
└── audit.log                           # Audit trail

/etc/systemd/system/                    # Systemd service
└── secure-memory-mcp.service           # Service definition

/etc/logrotate.d/                       # Log rotation
└── secure-memory-mcp                   # Rotation config

/etc/cron.d/                           # Scheduled tasks
└── secure-memory-backup               # Daily backup cron
```

## Key File Purposes

### Core Application
- **`server.py`** - Main MCP server with security enforcer, memory store, and handlers
- **`server_monitor.py`** - Optional process monitor for anomaly detection

### Configuration
- **`security_enforcer.json`** - Production config (rate limits, filters, sandboxing)
- **`sandbox_profiles.yaml`** - Docker/Firejail isolation profiles

### Testing
- **`test_security.py`** - Unit and integration tests for all components
- **`load_test.py`** - Performance and concurrency testing

### Deployment
- **`deploy.sh`** - Complete production deployment automation
- **`backup.sh`** - Database and config backup script

### Operations
- **`dashboard.html`** - Real-time monitoring interface
- **`status.sh`** - Quick health check script
- **`secure-memory-mcp.service`** - Systemd service with hardening

### Documentation
- **`README.md`** - Architecture and usage guide
- **`production_checklist.md`** - Step-by-step deployment
- **`fixes_and_improvements.md`** - Changes for production

## File Permissions (Production)

```bash
# Application files (root owned, world readable)
/opt/secure-memory-mcp/     755 root:root
/opt/secure-memory-mcp/**   644 root:root

# Configuration (root owned, group readable)
/etc/secure-memory-mcp/     750 root:secure-memory
/etc/secure-memory-mcp/*    640 root:secure-memory

# Data files (service user owned)
/var/lib/secure-memory-mcp/ 750 secure-memory:secure-memory
/var/lib/secure-memory-mcp/* 640 secure-memory:secure-memory

# Logs (service user owned)
/var/log/secure-memory-mcp/ 755 secure-memory:secure-memory
/var/log/secure-memory-mcp/* 644 secure-memory:secure-memory
```

## Configuration File Locations

1. **Development**: 
   - Config in `./config/`
   - Data in `~/.secure-memory-mcp/`

2. **Production**:
   - Config in `/etc/secure-memory-mcp/`
   - Data in `/var/lib/secure-memory-mcp/`
   - Logs in `/var/log/secure-memory-mcp/`

## Environment Variables

```bash
# Required
SECURE_MEMORY_CONFIG=/etc/secure-memory-mcp/security_enforcer.json

# Optional
PYTHONUNBUFFERED=1              # Better logging
HOME=/var/lib/secure-memory-mcp # Service home
SQLITE_TMPDIR=/var/lib/secure-memory-mcp/tmp  # SQLite temp
```

## Quick Reference

- **Start service**: `sudo systemctl start secure-memory-mcp`
- **View logs**: `sudo journalctl -u secure-memory-mcp -f`
- **Check status**: `sudo /opt/secure-memory-mcp/admin/status.sh`
- **Run backup**: `sudo /opt/secure-memory-mcp/backup.sh`
- **Edit config**: `sudo nano /etc/secure-memory-mcp/security_enforcer.json`
- **Test health**: `curl http://localhost:8080/health`

This structure ensures:
- Clear separation of code, config, and data
- Proper permissions for security
- Easy backup and monitoring
- Standard Linux service layout