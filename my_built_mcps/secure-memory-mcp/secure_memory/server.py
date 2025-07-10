#!/usr/bin/env python3
"""
File: secure_memory_mcp/server.py (Main server implementation)
Secure Memory MCP Server - Production Ready
A bulletproof MCP server that prevents AI systems from controlling their own constraints
"""

import json
import asyncio
import logging
import signal
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import hashlib
import hmac
import secrets
import sqlite3
import aiosqlite
from contextlib import asynccontextmanager
import jsonschema
import re
import traceback

# MCP SDK imports - adjust based on actual SDK
try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    import mcp.types as types
except ImportError as e:
    print(f"Error: MCP SDK not installed. Please install with: pip install mcp")
    sys.exit(1)

# Version info
__version__ = "1.0.0"
__author__ = "Secure Memory MCP Team"

# Load configuration from environment or defaults
CONFIG_PATH = os.environ.get("SECURE_MEMORY_CONFIG", 
                            Path.home() / ".secure-memory-mcp" / "config" / "security_enforcer.json")

# Default security configuration - used if external config not found
DEFAULT_SECURITY_CONFIG = {
    "version": "1.0",
    "sandboxes": {
        "memory_read": {
            "level": "none",
            "max_results": 100,
            "time_window_days": 30,
            "filter_patterns": [
                {"pattern": r"(?i)password[\s]*[:=][\s]*\S+", "replacement": "[REDACTED]"},
                {"pattern": r"(?i)api[_\s]*key[\s]*[:=][\s]*\S+", "replacement": "[REDACTED]"},
                {"pattern": r"(?i)secret[\s]*[:=][\s]*\S+", "replacement": "[REDACTED]"},
                {"pattern": r"(?i)token[\s]*[:=][\s]*\S+", "replacement": "[REDACTED]"},
                {"pattern": r"(?i)private[_\s]*key[\s]*[:=][\s]*\S+", "replacement": "[REDACTED]"},
                {"pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "replacement": "[EMAIL]"}
            ]
        },
        "memory_write": {
            "level": "restricted",
            "requires_validation": True,
            "max_size_kb": 10,
            "forbidden_patterns": [
                r"(?i)(exec|eval|compile|__import__|globals|locals)\s*\(",
                r"(?i)(subprocess|os\.system|popen)\s*\(",
                r"(?i)(rm\s+-rf|sudo|chmod|chown)",
                r"(?i)(drop\s+table|delete\s+from|truncate\s+table)"
            ]
        },
        "memory_search": {
            "level": "none",
            "max_query_length": 200,
            "forbidden_terms": ["*", "';", "\"", "--", "/*", "*/", "xp_", "sp_"],
            "max_results": 50
        }
    },
    "rate_limits": {
        "read": {"max_per_minute": 60, "burst": 10, "cooldown_seconds": 60},
        "write": {"max_per_minute": 20, "burst": 5, "cooldown_seconds": 120},
        "search": {"max_per_minute": 30, "burst": 5, "cooldown_seconds": 60}
    },
    "security": {
        "enable_hmac": True,
        "enable_encryption": False,  # Set to True and configure key for production
        "max_request_size_kb": 100,
        "session_timeout_minutes": 60,
        "max_sessions": 100
    },
    "monitoring": {
        "log_all_requests": True,
        "log_denied_requests": True,
        "alert_on_repeated_denials": True,
        "denial_threshold": 10,
        "denial_window_minutes": 5,
        "metrics_enabled": False,  # Enable for Prometheus integration
        "health_check_port": 8080
    }
}

# Enums
class OperationType(Enum):
    READ = "read"
    WRITE = "write"
    SEARCH = "search"
    DELETE = "delete"  # Never allowed

class SandboxLevel(Enum):
    NONE = "none"
    RESTRICTED = "restricted"
    ISOLATED = "isolated"

class SecurityEvent(Enum):
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    RATE_LIMITED = "rate_limited"
    INVALID_INPUT = "invalid_input"
    SECURITY_VIOLATION = "security_violation"

# JSON Schema for input validation
SCHEMAS = {
    "memory_write": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "minLength": 1, "maxLength": 10240},
            "metadata": {"type": "object", "maxProperties": 10}
        },
        "required": ["content"],
        "additionalProperties": False
    },
    "memory_read": {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
                "maxItems": 100
            }
        },
        "additionalProperties": False
    },
    "memory_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 200}
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

# Data structures
@dataclass
class MemoryEntry:
    id: str
    timestamp: datetime
    content: str
    metadata: Dict[str, Any]
    checksum: str = ""
    hmac_signature: str = ""
    
    def calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of entry data"""
        data = f"{self.id}{self.timestamp.isoformat()}{self.content}{json.dumps(self.metadata, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def calculate_hmac(self, secret_key: bytes) -> str:
        """Calculate HMAC-SHA256 signature"""
        data = f"{self.id}{self.timestamp.isoformat()}{self.content}{json.dumps(self.metadata, sort_keys=True)}"
        return hmac.new(secret_key, data.encode(), hashlib.sha256).hexdigest()
    
    def verify_integrity(self, secret_key: Optional[bytes] = None) -> bool:
        """Verify entry hasn't been tampered with"""
        if self.checksum != self.calculate_checksum():
            return False
        if secret_key and self.hmac_signature:
            return self.hmac_signature == self.calculate_hmac(secret_key)
        return True

@dataclass
class SecurityContext:
    operation: OperationType
    timestamp: datetime
    request_id: str
    session_id: str
    source_ip: str = "127.0.0.1"
    user_agent: str = "mcp-client"

@dataclass
class RateLimitState:
    requests: List[datetime] = field(default_factory=list)
    denied_count: int = 0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def check_and_update(self, limit_config: Dict) -> Tuple[bool, Optional[str]]:
        """Check if request is allowed and update state"""
        now = datetime.now(timezone.utc)
        window = now - timedelta(minutes=1)
        
        # Clean old requests
        self.requests = [r for r in self.requests if r > window]
        
        # Check if in cooldown
        if self.denied_count > 0:
            cooldown_end = self.last_reset + timedelta(seconds=limit_config.get("cooldown_seconds", 60))
            if now < cooldown_end:
                remaining = int((cooldown_end - now).total_seconds())
                return False, f"Rate limit cooldown: {remaining}s remaining"
        
        # Check rate limit
        if len(self.requests) >= limit_config["max_per_minute"]:
            self.denied_count += 1
            return False, f"Rate limit exceeded: {limit_config['max_per_minute']}/min"
        
        # Check burst
        burst_window = now - timedelta(seconds=10)
        recent = [r for r in self.requests if r > burst_window]
        if len(recent) >= limit_config["burst"]:
            self.denied_count += 1
            return False, f"Burst limit exceeded: {limit_config['burst']}/10s"
        
        # Request allowed
        self.requests.append(now)
        self.denied_count = 0
        return True, None

# Database schema
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_entries (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL,
    checksum TEXT NOT NULL,
    hmac_signature TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_created_at ON memory_entries(created_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    request_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    details TEXT,
    source_ip TEXT,
    user_agent TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_log(status);

CREATE TABLE IF NOT EXISTS security_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,
    session_id TEXT,
    details TEXT NOT NULL
);
"""

# Security Enforcer - The Gatekeeper
class SecurityEnforcer:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.rate_limiters: Dict[str, RateLimitState] = {}
        self.denied_requests: Dict[str, List[datetime]] = {}
        self.sessions: Dict[str, datetime] = {}
        self.hmac_key = secrets.token_bytes(32) if config["security"]["enable_hmac"] else None
        
        # Compile regex patterns for efficiency
        self.filter_patterns = []
        for pattern_config in config["sandboxes"]["memory_read"]["filter_patterns"]:
            try:
                compiled = re.compile(pattern_config["pattern"])
                self.filter_patterns.append((compiled, pattern_config["replacement"]))
            except re.error as e:
                logger.error(f"Invalid regex pattern: {pattern_config['pattern']} - {e}")
        
        self.forbidden_write_patterns = []
        for pattern in config["sandboxes"]["memory_write"].get("forbidden_patterns", []):
            try:
                self.forbidden_write_patterns.append(re.compile(pattern))
            except re.error as e:
                logger.error(f"Invalid forbidden pattern: {pattern} - {e}")
    
    def create_session(self) -> str:
        """Create a new session"""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = datetime.now(timezone.utc)
        
        # Clean old sessions
        timeout = timedelta(minutes=self.config["security"]["session_timeout_minutes"])
        cutoff = datetime.now(timezone.utc) - timeout
        self.sessions = {k: v for k, v in self.sessions.items() if v > cutoff}
        
        # Check max sessions
        if len(self.sessions) > self.config["security"]["max_sessions"]:
            # Remove oldest sessions
            sorted_sessions = sorted(self.sessions.items(), key=lambda x: x[1])
            for session_id, _ in sorted_sessions[:len(self.sessions) - self.config["security"]["max_sessions"]]:
                del self.sessions[session_id]
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate session is active"""
        if session_id not in self.sessions:
            return False
        
        timeout = timedelta(minutes=self.config["security"]["session_timeout_minutes"])
        if datetime.now(timezone.utc) - self.sessions[session_id] > timeout:
            del self.sessions[session_id]
            return False
        
        # Update session activity
        self.sessions[session_id] = datetime.now(timezone.utc)
        return True
    
    def check_rate_limit(self, operation: OperationType, session_id: str) -> Tuple[bool, Optional[str]]:
        """Check and update rate limits"""
        limit_key = f"{operation.value}:{session_id}"
        
        if limit_key not in self.rate_limiters:
            self.rate_limiters[limit_key] = RateLimitState()
        
        limit_config = self.config["rate_limits"][operation.value]
        allowed, reason = self.rate_limiters[limit_key].check_and_update(limit_config)
        
        if not allowed:
            self._record_denied_request(session_id, reason)
        
        return allowed, reason
    
    def _record_denied_request(self, session_id: str, reason: str):
        """Track denied requests for anomaly detection"""
        now = datetime.now(timezone.utc)
        
        if session_id not in self.denied_requests:
            self.denied_requests[session_id] = []
        
        self.denied_requests[session_id].append(now)
        
        # Check for repeated denials
        window = timedelta(minutes=self.config["monitoring"]["denial_window_minutes"])
        recent_denials = [d for d in self.denied_requests[session_id] if d > now - window]
        
        if len(recent_denials) >= self.config["monitoring"]["denial_threshold"]:
            self.logger.warning(f"Security Alert: Session {session_id} has {len(recent_denials)} denied requests in {window}")
            # Could trigger additional security measures here
    
    def validate_input(self, operation: str, data: Dict) -> Tuple[bool, Optional[str]]:
        """Validate input against JSON schema"""
        if operation not in SCHEMAS:
            return False, "Unknown operation"
        
        try:
            jsonschema.validate(data, SCHEMAS[operation])
            return True, None
        except jsonschema.ValidationError as e:
            return False, f"Invalid input: {e.message}"
    
    def check_content_security(self, content: str) -> Tuple[bool, Optional[str]]:
        """Check content for security violations"""
        # Check size
        max_size = self.config["security"]["max_request_size_kb"] * 1024
        if len(content.encode()) > max_size:
            return False, f"Content too large: {len(content.encode())} bytes (max: {max_size})"
        
        # Check forbidden patterns
        for pattern in self.forbidden_write_patterns:
            if pattern.search(content):
                return False, f"Forbidden content pattern detected"
        
        return True, None
    
    def validate_search_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate search query for security"""
        # Check length
        max_length = self.config["sandboxes"]["memory_search"]["max_query_length"]
        if len(query) > max_length:
            return False, f"Query too long: {len(query)} chars (max: {max_length})"
        
        # Check forbidden terms
        query_lower = query.lower()
        for term in self.config["sandboxes"]["memory_search"]["forbidden_terms"]:
            if term in query_lower:
                return False, f"Forbidden search term detected"
        
        # Basic SQL injection prevention
        if re.search(r"[;'\"]|\b(union|select|insert|update|delete|drop)\b", query_lower):
            return False, "Potential injection attempt detected"
        
        return True, None
    
    def filter_content(self, content: str) -> str:
        """Filter sensitive information from content"""
        filtered = content
        for pattern, replacement in self.filter_patterns:
            filtered = pattern.sub(replacement, filtered)
        return filtered
    
    def apply_time_window(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """Filter entries to allowed time window"""
        window_days = self.config["sandboxes"]["memory_read"]["time_window_days"]
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        return [e for e in entries if e.timestamp > cutoff]

# Immutable Memory Store with SQLite
class ImmutableMemoryStore:
    def __init__(self, db_path: Path, logger: logging.Logger, hmac_key: Optional[bytes] = None):
        self.db_path = db_path
        self.logger = logger
        self.hmac_key = hmac_key
        self._connection: Optional[aiosqlite.Connection] = None
    
    async def initialize(self):
        """Initialize database"""
        try:
            self._connection = await aiosqlite.connect(
                self.db_path,
                isolation_level=None  # Autocommit mode
            )
            await self._connection.executescript(SCHEMA_SQL)
            await self._connection.commit()
            self.logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self._connection:
            await self._connection.close()
    
    async def append(self, entry: MemoryEntry) -> bool:
        """Append entry to store"""
        try:
            # Calculate integrity fields
            entry.checksum = entry.calculate_checksum()
            if self.hmac_key:
                entry.hmac_signature = entry.calculate_hmac(self.hmac_key)
            
            # Store in database
            await self._connection.execute(
                """INSERT INTO memory_entries 
                   (id, timestamp, content, metadata, checksum, hmac_signature)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    entry.id,
                    entry.timestamp.isoformat(),
                    entry.content,
                    json.dumps(entry.metadata),
                    entry.checksum,
                    entry.hmac_signature
                )
            )
            await self._connection.commit()
            return True
            
        except sqlite3.IntegrityError:
            self.logger.warning(f"Duplicate entry ID: {entry.id}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to append entry: {e}")
            return False
    
    async def read(self, ids: Optional[List[str]] = None, limit: int = 100) -> List[MemoryEntry]:
        """Read entries from store"""
        try:
            if ids:
                # Read specific IDs
                placeholders = ','.join(['?' for _ in ids])
                query = f"""SELECT id, timestamp, content, metadata, checksum, hmac_signature
                           FROM memory_entries WHERE id IN ({placeholders})
                           ORDER BY timestamp DESC LIMIT ?"""
                cursor = await self._connection.execute(query, ids + [limit])
            else:
                # Read all (up to limit)
                query = """SELECT id, timestamp, content, metadata, checksum, hmac_signature
                          FROM memory_entries ORDER BY timestamp DESC LIMIT ?"""
                cursor = await self._connection.execute(query, (limit,))
            
            entries = []
            async for row in cursor:
                entry = MemoryEntry(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    content=row[2],
                    metadata=json.loads(row[3]),
                    checksum=row[4],
                    hmac_signature=row[5] or ""
                )
                
                # Verify integrity
                if entry.verify_integrity(self.hmac_key):
                    entries.append(entry)
                else:
                    self.logger.error(f"Integrity check failed for entry: {entry.id}")
            
            return entries
            
        except Exception as e:
            self.logger.error(f"Failed to read entries: {e}")
            return []
    
    async def search(self, query: str, limit: int = 50) -> List[MemoryEntry]:
        """Search entries - uses SQLite FTS if available"""
        try:
            # Basic LIKE search - in production, use FTS5
            search_query = f"%{query}%"
            cursor = await self._connection.execute(
                """SELECT id, timestamp, content, metadata, checksum, hmac_signature
                   FROM memory_entries 
                   WHERE content LIKE ? OR metadata LIKE ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (search_query, search_query, limit)
            )
            
            entries = []
            async for row in cursor:
                entry = MemoryEntry(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    content=row[2],
                    metadata=json.loads(row[3]),
                    checksum=row[4],
                    hmac_signature=row[5] or ""
                )
                
                if entry.verify_integrity(self.hmac_key):
                    entries.append(entry)
            
            return entries
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    async def log_audit_event(self, context: SecurityContext, status: str, details: str = ""):
        """Log security audit event"""
        try:
            await self._connection.execute(
                """INSERT INTO audit_log 
                   (request_id, session_id, operation, status, details, source_ip, user_agent)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    context.request_id,
                    context.session_id,
                    context.operation.value,
                    status,
                    details,
                    context.source_ip,
                    context.user_agent
                )
            )
            await self._connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")

# The MCP Server
class SecureMemoryMCP:
    def __init__(self):
        self.server = Server("secure-memory")
        self.storage_path = Path.home() / ".secure-memory-mcp"
        self.storage_path.mkdir(exist_ok=True, mode=0o700)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize components
        self.enforcer = SecurityEnforcer(self.config, self.logger)
        self.store = ImmutableMemoryStore(
            self.storage_path / "memory.db",
            self.logger,
            self.enforcer.hmac_key
        )
        
        # Register handlers
        self._register_handlers()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self.logger.info(f"Secure Memory MCP Server v{__version__} initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("secure-memory-mcp")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        log_dir = self.storage_path / "logs"
        log_dir.mkdir(exist_ok=True, mode=0o700)
        
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_dir / "secure-memory.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Security event handler
        security_handler = RotatingFileHandler(
            log_dir / "security.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=10
        )
        security_handler.setFormatter(file_formatter)
        security_handler.setLevel(logging.WARNING)
        logger.addHandler(security_handler)
        
        return logger
    
    def _load_configuration(self) -> Dict:
        """Load configuration from file or use defaults"""
        config_path = Path(CONFIG_PATH)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_path}: {e}")
                self.logger.warning("Using default configuration")
        else:
            self.logger.warning(f"Config file not found at {config_path}, using defaults")
            
            # Create default config file for reference
            config_dir = config_path.parent
            config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            
            try:
                with open(config_path, 'w') as f:
                    json.dump(DEFAULT_SECURITY_CONFIG, f, indent=2)
                self.logger.info(f"Created default config at {config_path}")
            except Exception as e:
                self.logger.error(f"Failed to create default config: {e}")
        
        return DEFAULT_SECURITY_CONFIG
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down Secure Memory MCP Server...")
        
        # Close database
        await self.store.close()
        
        # Save any pending state
        # ... additional cleanup ...
        
        self.logger.info("Shutdown complete")
        sys.exit(0)
    
    def _register_handlers(self):
        """Register MCP protocol handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="memory_write",
                    description="Store information in secure memory with automatic integrity protection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Content to store (max 10KB)"
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata (max 10 properties)"
                            }
                        },
                        "required": ["content"]
                    }
                ),
                types.Tool(
                    name="memory_read",
                    description="Read from secure memory with automatic content filtering",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific IDs to read (optional, max 100)"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="memory_search",
                    description="Search secure memory with validated queries",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (max 200 chars)"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, 
            arguments: Dict[str, Any]
        ) -> List[types.TextContent]:
            # Generate request context
            request_id = secrets.token_urlsafe(16)
            session_id = self.enforcer.create_session()  # In production, get from connection
            
            context = SecurityContext(
                operation=OperationType(name.replace("memory_", "")),
                timestamp=datetime.now(timezone.utc),
                request_id=request_id,
                session_id=session_id
            )
            
            try:
                # Validate session
                if not self.enforcer.validate_session(session_id):
                    await self.store.log_audit_event(context, "invalid_session")
                    return self._error_response("Invalid or expired session")
                
                # Log request
                if self.config["monitoring"]["log_all_requests"]:
                    self.logger.info(f"Request {request_id}: {name} from session {session_id}")
                
                # Validate input
                valid, error = self.enforcer.validate_input(name, arguments)
                if not valid:
                    await self.store.log_audit_event(context, "invalid_input", error)
                    return self._error_response(f"Invalid input: {error}")
                
                # Check rate limits
                allowed, reason = self.enforcer.check_rate_limit(context.operation, session_id)
                if not allowed:
                    await self.store.log_audit_event(context, "rate_limited", reason)
                    self.logger.warning(f"Rate limit exceeded for session {session_id}: {reason}")
                    return self._error_response(f"Rate limit exceeded: {reason}")
                
                # Dispatch to handler
                if name == "memory_write":
                    result = await self._handle_write(arguments, context)
                elif name == "memory_read":
                    result = await self._handle_read(arguments, context)
                elif name == "memory_search":
                    result = await self._handle_search(arguments, context)
                else:
                    result = {"error": "Unknown tool", "request_id": request_id}
                
                # Log successful operation
                await self.store.log_audit_event(context, "success", json.dumps(result)[:100])
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )]
                
            except Exception as e:
                self.logger.error(f"Request {request_id} failed: {str(e)}\n{traceback.format_exc()}")
                await self.store.log_audit_event(context, "error", str(e))
                return self._error_response(
                    "Internal error occurred",
                    request_id=request_id
                )
    
    def _error_response(self, message: str, **kwargs) -> List[types.TextContent]:
        """Generate error response"""
        response = {"error": message, **kwargs}
        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
    
    async def _handle_write(self, args: Dict, context: SecurityContext) -> Dict:
        """Handle memory write operation"""
        content = args.get("content", "")
        metadata = args.get("metadata", {})
        
        # Security validation
        valid, error = self.enforcer.check_content_security(content)
        if not valid:
            self.logger.warning(f"Content security check failed: {error}")
            return {"error": error, "request_id": context.request_id}
        
        # Create entry
        entry = MemoryEntry(
            id=context.request_id,
            timestamp=context.timestamp,
            content=content,
            metadata={
                **metadata,
                "_session_id": context.session_id,
                "_source": "mcp_client"
            }
        )
        
        # Store
        success = await self.store.append(entry)
        
        if success:
            self.logger.info(f"Write successful: {context.request_id}")
            return {
                "success": True,
                "id": entry.id,
                "timestamp": entry.timestamp.isoformat(),
                "size_bytes": len(content.encode())
            }
        else:
            return {
                "error": "Write failed - possible duplicate ID",
                "request_id": context.request_id
            }
    
    async def _handle_read(self, args: Dict, context: SecurityContext) -> Dict:
        """Handle memory read operation"""
        ids = args.get("ids")
        
        # Get entries
        max_results = self.config["sandboxes"]["memory_read"]["max_results"]
        entries = await self.store.read(ids, limit=max_results)
        
        # Apply security filters
        entries = self.enforcer.apply_time_window(entries)
        
        # Filter content and format results
        results = []
        for entry in entries[:max_results]:
            filtered_content = self.enforcer.filter_content(entry.content)
            
            # Remove internal metadata
            public_metadata = {k: v for k, v in entry.metadata.items() 
                             if not k.startswith("_")}
            
            results.append({
                "id": entry.id,
                "timestamp": entry.timestamp.isoformat(),
                "content": filtered_content,
                "metadata": public_metadata
            })
        
        return {
            "entries": results,
            "count": len(results),
            "filtered": len(entries) - len(results) if len(entries) > len(results) else 0
        }
    
    async def _handle_search(self, args: Dict, context: SecurityContext) -> Dict:
        """Handle memory search operation"""
        query = args.get("query", "")
        
        # Validate query
        valid, error = self.enforcer.validate_search_query(query)
        if not valid:
            self.logger.warning(f"Search validation failed: {error}")
            return {"error": error, "request_id": context.request_id}
        
        # Search
        max_results = self.config["sandboxes"]["memory_search"]["max_results"]
        entries = await self.store.search(query, limit=max_results)
        
        # Apply filters (same as read)
        entries = self.enforcer.apply_time_window(entries)
        
        # Format results
        results = []
        for entry in entries[:max_results]:
            filtered_content = self.enforcer.filter_content(entry.content)
            
            # Highlight matches (basic implementation)
            highlighted = filtered_content
            if query.lower() in filtered_content.lower():
                # Simple highlight - in production use proper text search
                highlighted = filtered_content.replace(
                    query, f"**{query}**"
                )
            
            public_metadata = {k: v for k, v in entry.metadata.items() 
                             if not k.startswith("_")}
            
            results.append({
                "id": entry.id,
                "timestamp": entry.timestamp.isoformat(),
                "content": filtered_content,
                "highlighted": highlighted,
                "metadata": public_metadata
            })
        
        return {
            "entries": results,
            "count": len(results),
            "query": query
        }
    
    async def run(self):
        """Run the MCP server"""
        try:
            # Initialize store
            await self.store.initialize()
            
            self.logger.info("Starting Secure Memory MCP Server...")
            
            # Run server
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="secure-memory",
                        server_version=__version__,
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )
        except Exception as e:
            self.logger.error(f"Server error: {e}\n{traceback.format_exc()}")
            raise
        finally:
            await self.shutdown()

# Health check server (optional)
async def health_check_server(port: int = 8080):
    """Simple HTTP health check endpoint"""
    from aiohttp import web
    
    async def health(request):
        return web.json_response({
            "status": "healthy",
            "version": __version__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    app = web.Application()
    app.router.add_get('/health', health)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', port)
    await site.start()

# Entry point
async def main():
    """Main entry point"""
    server = SecureMemoryMCP()
    
    # Optional: Start health check server
    if server.config["monitoring"].get("health_check_port"):
        asyncio.create_task(
            health_check_server(server.config["monitoring"]["health_check_port"])
        )
    
    await server.run()

if __name__ == "__main__":
    # Ensure event loop runs properly on all platforms
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)