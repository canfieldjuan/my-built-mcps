#!/usr/bin/env python3
# File: universal_api_gateway_v2.py
"""
Production-Ready Universal API Gateway MCP Server - FastMCP 2.0
Bulletproof implementation for heavy traffic and indefinite runtime.
"""

import asyncio
import json
import time
import hashlib
import base64
import os
import signal
import sys
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import aiofiles
import aiosqlite
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
from fastmcp import FastMCP, Context
from urllib.parse import urljoin, urlparse, urlencode
from contextlib import asynccontextmanager
import logging
import logging.handlers
from collections import defaultdict
import weakref
import gc
from enum import Enum
import ssl

# ===== PRODUCTION CONFIGURATION =====

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class TransportType(str, Enum):
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"

@dataclass
class ProductionConfig:
    """Production configuration with environment variable support"""
    # Server Configuration
    host: str = "127.0.0.1"
    port: int = 8000
    transport: TransportType = TransportType.STDIO
    log_level: LogLevel = LogLevel.INFO
    
    # Database Configuration
    cache_db_path: str = "data/cache.db"
    max_cache_size: int = 50000
    cache_cleanup_interval: int = 300  # 5 minutes
    
    # HTTP Configuration
    max_connections: int = 1000
    max_connections_per_host: int = 100
    connection_timeout: int = 30
    request_timeout: int = 120
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_response_size: int = 50 * 1024 * 1024  # 50MB
    
    # Rate Limiting
    default_rate_limit: int = 1000  # requests per minute
    rate_limit_cleanup_interval: int = 3600  # 1 hour
    
    # Circuit Breaker
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: int = 60
    circuit_breaker_recovery_timeout: int = 300
    
    # Health Monitoring
    health_check_interval: int = 60
    memory_threshold_mb: int = 1024
    
    # Security
    max_redirects: int = 3
    verify_ssl: bool = True
    allowed_schemes: Set[str] = Field(default_factory=lambda: {"http", "https"})
    
    @classmethod
    def from_env(cls) -> 'ProductionConfig':
        """Load configuration from environment variables"""
        return cls(
            host=os.getenv("API_GATEWAY_HOST", "127.0.0.1"),
            port=int(os.getenv("API_GATEWAY_PORT", "8000")),
            transport=TransportType(os.getenv("API_GATEWAY_TRANSPORT", "stdio")),
            log_level=LogLevel(os.getenv("API_GATEWAY_LOG_LEVEL", "INFO")),
            cache_db_path=os.getenv("API_GATEWAY_CACHE_DB", "data/cache.db"),
            max_cache_size=int(os.getenv("API_GATEWAY_MAX_CACHE_SIZE", "50000")),
            max_connections=int(os.getenv("API_GATEWAY_MAX_CONNECTIONS", "1000")),
            max_connections_per_host=int(os.getenv("API_GATEWAY_MAX_CONN_PER_HOST", "100")),
            connection_timeout=int(os.getenv("API_GATEWAY_CONN_TIMEOUT", "30")),
            request_timeout=int(os.getenv("API_GATEWAY_REQ_TIMEOUT", "120")),
            max_request_size=int(os.getenv("API_GATEWAY_MAX_REQ_SIZE", str(10 * 1024 * 1024))),
            verify_ssl=os.getenv("API_GATEWAY_VERIFY_SSL", "true").lower() == "true",
        )

# Configure production logging
def setup_production_logging(config: ProductionConfig):
    """Setup production-grade logging with rotation and structured output"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.log_level.value))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "api_gateway.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "api_gateway_errors.log",
        maxBytes=10*1024*1024,
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)

# ===== ENHANCED PYDANTIC MODELS =====

class AuthConfig(BaseModel):
    """Enhanced authentication configuration with validation"""
    location: Optional[str] = Field("header", regex="^(header|query|body)$")
    key: str = Field(..., min_length=1, max_length=255)
    value: str = Field(..., min_length=1)
    username: Optional[str] = Field(None, min_length=1, max_length=255)
    password: Optional[str] = Field(None, min_length=1)
    token: Optional[str] = Field(None, min_length=1)
    
    class Config:
        extra = "forbid"

class APIServiceConfig(BaseModel):
    """Enhanced API service configuration with production validation"""
    name: str = Field(..., min_length=1, max_length=100, regex="^[a-zA-Z0-9_-]+$")
    base_url: str = Field(..., min_length=1)
    auth_type: str = Field(..., regex="^(none|api_key|bearer|oauth|basic)$")
    auth_config: Optional[AuthConfig] = None
    default_headers: Dict[str, str] = Field(default_factory=dict)
    rate_limit: Optional[int] = Field(None, ge=1, le=10000)
    timeout: int = Field(30, ge=1, le=300)
    retry_attempts: int = Field(3, ge=1, le=10)
    cache_ttl: int = Field(300, ge=0, le=86400)
    circuit_breaker_enabled: bool = Field(True)
    health_check_path: Optional[str] = None
    
    @validator('base_url')
    def validate_base_url(cls, v):
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError('Invalid base URL format')
        if parsed.scheme not in ['http', 'https']:
            raise ValueError('Only HTTP and HTTPS schemes are allowed')
        return v.rstrip('/')
    
    class Config:
        extra = "forbid"

class EndpointConfig(BaseModel):
    """Enhanced endpoint configuration with validation"""
    name: str = Field(..., min_length=1, max_length=100, regex="^[a-zA-Z0-9_-]+$")
    service: str = Field(..., min_length=1, max_length=100)
    path: str = Field(..., min_length=1)
    method: str = Field(..., regex="^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)$")
    description: str = Field(..., min_length=1, max_length=1000)
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)
    response_schema: Optional[Dict[str, Any]] = None
    cache_enabled: bool = True
    requires_auth: bool = True
    tags: List[str] = Field(default_factory=list)
    max_request_size: Optional[int] = Field(None, ge=1)
    
    @validator('path')
    def validate_path(cls, v):
        if not v.startswith('/'):
            v = '/' + v
        return v
    
    class Config:
        extra = "forbid"

# ===== CIRCUIT BREAKER IMPLEMENTATION =====

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerStats:
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0
    state: CircuitState = CircuitState.CLOSED

class CircuitBreaker:
    """Production-grade circuit breaker for service protection"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.services: Dict[str, CircuitBreakerStats] = {}
        self.lock = asyncio.Lock()
        
    async def call(self, service_name: str, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        async with self.lock:
            stats = self.services.setdefault(service_name, CircuitBreakerStats())
            
            # Check circuit state
            if stats.state == CircuitState.OPEN:
                if time.time() - stats.last_failure_time > self.config.circuit_breaker_recovery_timeout:
                    stats.state = CircuitState.HALF_OPEN
                    logging.info(f"Circuit breaker for {service_name} entering half-open state")
                else:
                    raise Exception(f"Circuit breaker OPEN for service {service_name}")
            
        try:
            result = await func(*args, **kwargs)
            
            async with self.lock:
                stats.success_count += 1
                if stats.state == CircuitState.HALF_OPEN:
                    stats.state = CircuitState.CLOSED
                    stats.failure_count = 0
                    logging.info(f"Circuit breaker for {service_name} closed")
                    
            return result
            
        except Exception as e:
            async with self.lock:
                stats.failure_count += 1
                stats.last_failure_time = time.time()
                
                if stats.failure_count >= self.config.circuit_breaker_failure_threshold:
                    stats.state = CircuitState.OPEN
                    logging.warning(f"Circuit breaker OPENED for service {service_name}")
                    
            raise

# ===== ENHANCED CACHE MANAGER =====

class ProductionCacheManager:
    """Production-grade cache with monitoring and size limits"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.db_path = Path(config.cache_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.lock = asyncio.Lock()
        self._cleanup_task = None
        self._monitoring_task = None
        self.stats = {
            "hits": 0, "misses": 0, "sets": 0, "evictions": 0,
            "current_size": 0, "memory_usage": 0
        }
        
    async def initialize(self):
        """Initialize cache with production settings"""
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                # Optimize SQLite for production
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA synchronous=NORMAL")
                await db.execute("PRAGMA cache_size=10000")
                await db.execute("PRAGMA temp_store=MEMORY")
                await db.execute("PRAGMA mmap_size=268435456")  # 256MB
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        data BLOB NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        hit_count INTEGER DEFAULT 0,
                        data_size INTEGER NOT NULL,
                        last_accessed REAL NOT NULL
                    )
                """)
                
                # Create optimized indexes
                await db.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache(last_accessed)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_hit_count ON cache(hit_count)")
                await db.commit()
                
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            self._monitoring_task = asyncio.create_task(self._monitor_performance())
            
            logging.info(f"Production cache initialized at {self.db_path}")
            
        except Exception as e:
            logging.error(f"Failed to initialize cache: {e}")
            raise

    async def get(self, key: str) -> Optional[bytes]:
        """Get cached value with performance tracking"""
        async with self.lock:
            try:
                async with aiosqlite.connect(str(self.db_path)) as db:
                    now = time.time()
                    async with db.execute(
                        """SELECT data, hit_count FROM cache 
                           WHERE key = ? AND expires_at > ?""",
                        (key, now)
                    ) as cursor:
                        result = await cursor.fetchone()
                        
                        if result:
                            # Update access statistics
                            await db.execute(
                                """UPDATE cache 
                                   SET hit_count = hit_count + 1, last_accessed = ? 
                                   WHERE key = ?""",
                                (now, key)
                            )
                            await db.commit()
                            self.stats["hits"] += 1
                            return result[0]
                        
                        self.stats["misses"] += 1
                        return None
                        
            except Exception as e:
                logging.error(f"Cache get error for key {key}: {e}")
                self.stats["misses"] += 1
                return None

    async def set(self, key: str, data: bytes, ttl: int):
        """Set cached value with size management"""
        if len(data) > self.config.max_request_size:
            logging.warning(f"Skipping cache set for oversized data: {len(data)} bytes")
            return
            
        now = time.time()
        expires_at = now + ttl
        data_size = len(data)
        
        async with self.lock:
            try:
                async with aiosqlite.connect(str(self.db_path)) as db:
                    # Check and manage cache size
                    await self._enforce_size_limits(db)
                    
                    await db.execute(
                        """INSERT OR REPLACE INTO cache 
                           (key, data, created_at, expires_at, hit_count, data_size, last_accessed) 
                           VALUES (?, ?, ?, ?, 0, ?, ?)""",
                        (key, data, now, expires_at, data_size, now)
                    )
                    await db.commit()
                    self.stats["sets"] += 1
                    
            except Exception as e:
                logging.error(f"Cache set error for key {key}: {e}")

    async def _enforce_size_limits(self, db: aiosqlite.Connection):
        """Enforce cache size limits using LRU eviction"""
        # Get current cache size
        async with db.execute("SELECT COUNT(*), COALESCE(SUM(data_size), 0) FROM cache") as cursor:
            count, total_size = await cursor.fetchone()
            
        if count >= self.config.max_cache_size:
            # Remove 20% of least recently used items
            evict_count = max(1, count // 5)
            await db.execute(
                """DELETE FROM cache WHERE key IN (
                    SELECT key FROM cache 
                    ORDER BY last_accessed ASC, hit_count ASC 
                    LIMIT ?
                )""",
                (evict_count,)
            )
            self.stats["evictions"] += evict_count
            logging.info(f"Cache evicted {evict_count} items")

    async def _periodic_cleanup(self):
        """Periodic cleanup with error handling"""
        while True:
            try:
                await asyncio.sleep(self.config.cache_cleanup_interval)
                
                async with self.lock:
                    async with aiosqlite.connect(str(self.db_path)) as db:
                        now = time.time()
                        result = await db.execute(
                            "DELETE FROM cache WHERE expires_at <= ?", (now,)
                        )
                        await db.commit()
                        
                        if result.rowcount > 0:
                            logging.info(f"Cache cleanup removed {result.rowcount} expired entries")
                            
                        # Vacuum occasionally for space reclamation
                        if time.time() % 3600 < self.config.cache_cleanup_interval:  # Once per hour
                            await db.execute("VACUUM")
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Cache cleanup error: {e}")

    async def _monitor_performance(self):
        """Monitor cache performance and health"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                async with aiosqlite.connect(str(self.db_path)) as db:
                    async with db.execute(
                        "SELECT COUNT(*), COALESCE(SUM(data_size), 0) FROM cache"
                    ) as cursor:
                        count, size = await cursor.fetchone()
                        
                self.stats["current_size"] = count
                self.stats["memory_usage"] = size
                
                # Log performance metrics
                total_requests = self.stats["hits"] + self.stats["misses"]
                hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
                
                if total_requests % 1000 == 0 and total_requests > 0:
                    logging.info(
                        f"Cache performance: {hit_rate:.2%} hit rate, "
                        f"{count} entries, {size/1024/1024:.1f}MB"
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Cache monitoring error: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            **self.stats,
            "hit_rate": self.stats["hits"] / (self.stats["hits"] + self.stats["misses"]) 
                       if (self.stats["hits"] + self.stats["misses"]) > 0 else 0
        }

    async def close(self):
        """Clean shutdown of cache manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()
            
        # Wait for tasks to complete
        tasks = [t for t in [self._cleanup_task, self._monitoring_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

# ===== ENHANCED RATE LIMITER =====

class ProductionRateLimiter:
    """Production rate limiter with client tracking and cleanup"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.buckets: Dict[str, Dict] = {}
        self.client_buckets: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        self.lock = asyncio.Lock()
        self._cleanup_task = None
        
    async def initialize(self):
        """Initialize rate limiter"""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
    async def can_request(self, service: str, limit: int, client_id: Optional[str] = None) -> bool:
        """Check rate limit with optional per-client limiting"""
        async with self.lock:
            now = time.time()
            
            # Service-level rate limiting
            service_bucket = await self._get_or_create_bucket(
                self.buckets, service, limit, now
            )
            
            if not self._check_bucket(service_bucket, now, limit):
                return False
                
            # Client-level rate limiting if specified
            if client_id:
                client_limit = min(limit // 10, 100)  # 10% of service limit per client
                client_bucket = await self._get_or_create_bucket(
                    self.client_buckets[service], client_id, client_limit, now
                )
                
                if not self._check_bucket(client_bucket, now, client_limit):
                    return False
                    
            return True
            
    async def _get_or_create_bucket(self, bucket_dict: Dict, key: str, limit: int, now: float) -> Dict:
        """Get or create a rate limit bucket"""
        if key not in bucket_dict:
            bucket_dict[key] = {
                "tokens": limit,
                "last_refill": now,
                "limit": limit,
                "last_access": now
            }
        return bucket_dict[key]
        
    def _check_bucket(self, bucket: Dict, now: float, limit: int) -> bool:
        """Check and update bucket tokens"""
        bucket["last_access"] = now
        
        # Refill tokens
        time_passed = now - bucket["last_refill"]
        tokens_to_add = (time_passed / 60) * limit
        bucket["tokens"] = min(limit, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
        
        # Check if request allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False
        
    async def _periodic_cleanup(self):
        """Clean up inactive buckets"""
        while True:
            try:
                await asyncio.sleep(self.config.rate_limit_cleanup_interval)
                
                async with self.lock:
                    now = time.time()
                    cutoff = now - (2 * self.config.rate_limit_cleanup_interval)
                    
                    # Clean service buckets
                    to_remove = [
                        service for service, bucket in self.buckets.items()
                        if bucket.get("last_access", 0) < cutoff
                    ]
                    for service in to_remove:
                        del self.buckets[service]
                        
                    # Clean client buckets
                    for service in list(self.client_buckets.keys()):
                        client_buckets = self.client_buckets[service]
                        to_remove_clients = [
                            client for client, bucket in client_buckets.items()
                            if bucket.get("last_access", 0) < cutoff
                        ]
                        for client in to_remove_clients:
                            del client_buckets[client]
                            
                        if not client_buckets:
                            del self.client_buckets[service]
                            
                    total_removed = len(to_remove) + sum(len(to_remove_clients) for to_remove_clients in [])
                    if total_removed > 0:
                        logging.info(f"Rate limiter cleaned up {total_removed} inactive buckets")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Rate limiter cleanup error: {e}")
                
    async def close(self):
        """Clean shutdown"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

# ===== PRODUCTION API GATEWAY =====

class ProductionAPIGateway:
    """Production-ready Universal API Gateway with full monitoring"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.services: Dict[str, APIServiceConfig] = {}
        self.endpoints: Dict[str, EndpointConfig] = {}
        self.cache = ProductionCacheManager(config)
        self.rate_limiter = ProductionRateLimiter(config)
        self.circuit_breaker = CircuitBreaker(config)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # State management
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._health_task = None
        self._config_watcher_task = None
        
        # Monitoring
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Configuration file paths
        self.services_file = Path("config/api_services.json")
        self.endpoints_file = Path("config/api_endpoints.json")
        
    async def initialize(self):
        """Initialize gateway with full error handling"""
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                logging.info("Initializing Production API Gateway...")
                
                # Create SSL context
                ssl_context = ssl.create_default_context()
                if not self.config.verify_ssl:
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    
                # Create optimized connector
                connector = aiohttp.TCPConnector(
                    limit=self.config.max_connections,
                    limit_per_host=self.config.max_connections_per_host,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    enable_cleanup_closed=True,
                    ssl=ssl_context,
                    keepalive_timeout=30,
                    limit_max_redirects=self.config.max_redirects
                )
                
                # Create session with production settings
                timeout = aiohttp.ClientTimeout(
                    total=self.config.request_timeout,
                    connect=self.config.connection_timeout,
                    sock_read=30
                )
                
                self.session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={
                        "User-Agent": "Universal-API-Gateway/2.0-Production",
                        "Accept": "*/*",
                        "Accept-Encoding": "gzip, deflate"
                    },
                    read_bufsize=65536,
                    max_line_size=8192,
                    max_field_size=8192
                )
                
                # Initialize components
                await self.cache.initialize()
                await self.rate_limiter.initialize()
                await self._load_configurations()
                await self._validate_configurations()
                
                # Start monitoring tasks
                self._health_task = asyncio.create_task(self._health_monitor())
                self._config_watcher_task = asyncio.create_task(self._config_file_watcher())
                
                self._initialized = True
                logging.info("Production API Gateway initialized successfully")
                
            except Exception as e:
                logging.error(f"Failed to initialize gateway: {e}")
                await self.close()
                raise

    async def _load_configurations(self):
        """Load configurations with error handling"""
        # Ensure config directories exist
        self.services_file.parent.mkdir(parents=True, exist_ok=True)
        self.endpoints_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load services
            if self.services_file.exists():
                async with aiofiles.open(self.services_file, "r") as f:
                    content = await f.read()
                    services_data = json.loads(content)
                    
                    for name, config_data in services_data.items():
                        try:
                            config = APIServiceConfig(name=name, **config_data)
                            self.services[name] = config
                            logging.info(f"Loaded service: {name}")
                        except Exception as e:
                            logging.error(f"Invalid service config {name}: {e}")
                            
            # Load endpoints
            if self.endpoints_file.exists():
                async with aiofiles.open(self.endpoints_file, "r") as f:
                    content = await f.read()
                    endpoints_data = json.loads(content)
                    
                    for name, config_data in endpoints_data.items():
                        try:
                            config = EndpointConfig(name=name, **config_data)
                            self.endpoints[name] = config
                            logging.info(f"Loaded endpoint: {name}")
                        except Exception as e:
                            logging.error(f"Invalid endpoint config {name}: {e}")
                            
        except Exception as e:
            logging.error(f"Configuration loading error: {e}")
            raise

    async def _validate_configurations(self):
        """Comprehensive configuration validation"""
        errors = []
        
        # Check endpoint-service relationships
        for endpoint_name, endpoint_config in self.endpoints.items():
            if endpoint_config.service not in self.services:
                errors.append(f"Endpoint '{endpoint_name}' references unknown service '{endpoint_config.service}'")
                
            # Validate endpoint paths
            try:
                urlparse(endpoint_config.path)
            except Exception as e:
                errors.append(f"Invalid path in endpoint '{endpoint_name}': {e}")
                
        # Check service URLs
        for service_name, service_config in self.services.items():
            try:
                parsed = urlparse(service_config.base_url)
                if parsed.scheme not in self.config.allowed_schemes:
                    errors.append(f"Service '{service_name}' uses disallowed scheme: {parsed.scheme}")
            except Exception as e:
                errors.append(f"Invalid URL in service '{service_name}': {e}")
                
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        logging.info(f"Configuration validated: {len(self.services)} services, {len(self.endpoints)} endpoints")

    def _build_url(self, base_url: str, path: str, params: Dict[str, Any] = None, 
                   auth_config: Optional[AuthConfig] = None) -> str:
        """Build URL with comprehensive validation"""
        try:
            # Validate and parse base URL
            parsed_base = urlparse(base_url)
            if not parsed_base.scheme or not parsed_base.netloc:
                raise ValueError(f"Invalid base URL: {base_url}")
                
            if parsed_base.scheme not in self.config.allowed_schemes:
                raise ValueError(f"Disallowed URL scheme: {parsed_base.scheme}")
                
            # Clean and join path
            clean_path = path.lstrip('/')
            url = urljoin(base_url.rstrip('/') + '/', clean_path)
            
            # Build query parameters
            query_params = {}
            if params:
                for k, v in params.items():
                    if v is not None and len(str(v)) <= 1000:  # Reasonable param size limit
                        query_params[k] = str(v)
                        
            # Add auth query parameter if needed
            if auth_config and auth_config.location == "query":
                query_params[auth_config.key] = auth_config.value
                
            if query_params:
                url = f"{url}?{urlencode(query_params, safe='', quote_via=urlencode)}"
                
            # Final validation
            if len(url) > 2048:  # RFC 2616 recommended limit
                raise ValueError("URL too long")
                
            return url
            
        except Exception as e:
            raise ValueError(f"URL building failed: {e}")

    def _prepare_auth_headers(self, service_config: APIServiceConfig) -> Dict[str, str]:
        """Prepare authentication headers with validation"""
        headers = {}
        auth_type = service_config.auth_type
        auth_config = service_config.auth_config
        
        if not auth_config or auth_type == "none":
            return headers
            
        try:
            if auth_type == "api_key":
                if auth_config.location == "header":
                    headers[auth_config.key] = auth_config.value
            elif auth_type == "bearer":
                token = auth_config.token or auth_config.value
                headers["Authorization"] = f"Bearer {token}"
            elif auth_type == "basic":
                if auth_config.username and auth_config.password:
                    credentials = f"{auth_config.username}:{auth_config.password}"
                    encoded = base64.b64encode(credentials.encode()).decode()
                    headers["Authorization"] = f"Basic {encoded}"
            elif auth_type == "oauth":
                token = auth_config.token or auth_config.value
                headers["Authorization"] = f"Bearer {token}"
                
        except Exception as e:
            logging.error(f"Auth header preparation failed: {e}")
            
        return headers

    def _generate_cache_key(self, service: str, endpoint: str, params: Dict[str, Any], 
                          headers: Dict[str, str]) -> str:
        """Generate cache key with security considerations"""
        # Filter out sensitive headers
        safe_headers = {
            k: v for k, v in headers.items() 
            if k.lower() not in ['authorization', 'x-api-key', 'cookie', 'x-auth-token']
        }
        
        key_data = {
            "service": service,
            "endpoint": endpoint,
            "params": params,
            "headers": safe_headers
        }
        
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    async def make_api_request(
        self,
        service_name: str,
        endpoint_name: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        override_method: Optional[str] = None,
        client_id: Optional[str] = None,
        ctx: Optional[Context] = None
    ) -> Dict[str, Any]:
        """Production-grade API request with full monitoring"""
        
        request_start = time.time()
        self.request_count += 1
        
        try:
            if not self._initialized:
                await self.initialize()

            # Validate inputs
            if service_name not in self.services:
                raise ValueError(f"Service '{service_name}' not configured")
                
            if endpoint_name not in self.endpoints:
                raise ValueError(f"Endpoint '{endpoint_name}' not configured")

            service_config = self.services[service_name]
            endpoint_config = self.endpoints[endpoint_name]
            
            # Check rate limiting
            rate_limit = service_config.rate_limit or self.config.default_rate_limit
            if not await self.rate_limiter.can_request(service_name, rate_limit, client_id):
                raise Exception(f"Rate limit exceeded for {service_name}")

            # Prepare request components
            method = (override_method or endpoint_config.method).upper()
            headers = {**service_config.default_headers}
            
            if extra_headers:
                headers.update(extra_headers)
                
            # Add authentication
            if endpoint_config.requires_auth:
                auth_headers = self._prepare_auth_headers(service_config)
                headers.update(auth_headers)

            # Handle parameters and body
            request_params = params or {}
            if method == "GET":
                url = self._build_url(
                    service_config.base_url, 
                    endpoint_config.path, 
                    request_params, 
                    service_config.auth_config
                )
                json_body = None
            else:
                url = self._build_url(
                    service_config.base_url, 
                    endpoint_config.path, 
                    auth_config=service_config.auth_config
                )
                json_body = {**(body or {}), **request_params}
                
                # Check request size
                if json_body:
                    body_size = len(json.dumps(json_body).encode('utf-8'))
                    max_size = endpoint_config.max_request_size or self.config.max_request_size
                    if body_size > max_size:
                        raise ValueError(f"Request body too large: {body_size} > {max_size}")

            # Check cache for GET requests
            cache_key = None
            if endpoint_config.cache_enabled and method == "GET":
                cache_key = self._generate_cache_key(service_name, endpoint_name, request_params, headers)
                cached_response = await self.cache.get(cache_key)
                if cached_response:
                    if ctx:
                        await ctx.info(f"Cache hit for {service_name}/{endpoint_name}")
                    return json.loads(cached_response.decode('utf-8'))

            # Progress reporting
            if ctx:
                await ctx.info(f"Making {method} request to {service_name}/{endpoint_name}")

            # Execute request with circuit breaker
            async def make_request():
                async with self.session.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    timeout=aiohttp.ClientTimeout(total=service_config.timeout)
                ) as response:
                    
                    # Check response size
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.config.max_response_size:
                        raise Exception(f"Response too large: {content_length}")
                    
                    # Read response content with size limit
                    content = await response.read()
                    if len(content) > self.config.max_response_size:
                        raise Exception(f"Response content too large: {len(content)}")
                    
                    # Process response
                    response_data = {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "url": str(response.url),
                        "method": method,
                        "timestamp": datetime.utcnow().isoformat(),
                        "response_time_ms": round((time.time() - request_start) * 1000, 2)
                    }
                    
                    # Handle different content types
                    content_type = response.headers.get("content-type", "")
                    try:
                        if "application/json" in content_type:
                            response_data["data"] = json.loads(content.decode('utf-8'))
                        elif "text/" in content_type:
                            response_data["data"] = content.decode('utf-8')
                        else:
                            response_data["data"] = {
                                "content_type": content_type,
                                "content_length": len(content),
                                "note": "Binary data not included in response"
                            }
                    except Exception as e:
                        logging.warning(f"Response parsing error: {e}")
                        response_data["data"] = f"Response parsing error: {e}"
                    
                    return response_data, content

            # Execute with circuit breaker and retry
            if service_config.circuit_breaker_enabled:
                response_data, content = await self.circuit_breaker.call(
                    service_name, self._make_request_with_retry, 
                    make_request, service_config.retry_attempts, ctx
                )
            else:
                response_data, content = await self._make_request_with_retry(
                    make_request, service_config.retry_attempts, ctx
                )
            
            # Cache successful GET requests
            if (cache_key and response_data["status"] == 200 and 
                endpoint_config.cache_enabled and len(content) <= self.config.max_request_size):
                try:
                    cache_data = json.dumps(response_data).encode('utf-8')
                    await self.cache.set(cache_key, cache_data, service_config.cache_ttl)
                except Exception as e:
                    logging.warning(f"Cache set error: {e}")
            
            if ctx:
                await ctx.info(f"Request completed with status {response_data['status']}")
            
            return response_data
            
        except Exception as e:
            self.error_count += 1
            if ctx:
                await ctx.error(f"Request failed: {str(e)}")
            logging.error(f"API request error: {e}", extra={
                "service": service_name,
                "endpoint": endpoint_name,
                "error_type": type(e).__name__
            })
            raise

    async def _make_request_with_retry(self, request_func, max_retries: int, ctx: Optional[Context]):
        """Execute request with exponential backoff retry"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return await request_func()
                
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)  # Cap at 30 seconds
                    if ctx:
                        await ctx.info(f"Request timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    
            except aiohttp.ClientError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    if ctx:
                        await ctx.info(f"Client error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                # Don't retry on non-retryable errors
                raise e
                
        raise last_error

    async def _health_monitor(self):
        """Monitor system health and performance"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check memory usage
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb > self.config.memory_threshold_mb:
                    logging.warning(f"High memory usage: {memory_mb:.1f}MB")
                    
                    # Force garbage collection
                    gc.collect()
                    
                # Log health metrics
                uptime = time.time() - self.start_time
                error_rate = self.error_count / max(self.request_count, 1)
                
                logging.info(f"Health check: {self.request_count} requests, "
                           f"{error_rate:.2%} error rate, {memory_mb:.1f}MB memory, "
                           f"{uptime/3600:.1f}h uptime")
                           
                # Check service health
                await self._check_service_health()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Health monitor error: {e}")

    async def _check_service_health(self):
        """Check health of configured services"""
        for service_name, service_config in self.services.items():
            if service_config.health_check_path:
                try:
                    health_url = self._build_url(
                        service_config.base_url, 
                        service_config.health_check_path
                    )
                    
                    async with self.session.get(
                        health_url, 
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status != 200:
                            logging.warning(f"Service {service_name} health check failed: {response.status}")
                            
                except Exception as e:
                    logging.warning(f"Service {service_name} health check error: {e}")

    async def _config_file_watcher(self):
        """Watch for configuration file changes and reload"""
        last_services_mtime = 0
        last_endpoints_mtime = 0
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check services file
                if self.services_file.exists():
                    mtime = self.services_file.stat().st_mtime
                    if mtime > last_services_mtime:
                        logging.info("Services configuration changed, reloading...")
                        await self._load_configurations()
                        await self._validate_configurations()
                        last_services_mtime = mtime
                        
                # Check endpoints file
                if self.endpoints_file.exists():
                    mtime = self.endpoints_file.stat().st_mtime
                    if mtime > last_endpoints_mtime:
                        logging.info("Endpoints configuration changed, reloading...")
                        await self._load_configurations()
                        await self._validate_configurations()
                        last_endpoints_mtime = mtime
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Config watcher error: {e}")

    async def save_configurations(self):
        """Save configurations with atomic writes"""
        try:
            # Save services with atomic write
            services_data = {name: config.dict() for name, config in self.services.items()}
            temp_file = self.services_file.with_suffix('.tmp')
            
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(services_data, indent=2))
            temp_file.replace(self.services_file)
            
            # Save endpoints with atomic write
            endpoints_data = {name: config.dict() for name, config in self.endpoints.items()}
            temp_file = self.endpoints_file.with_suffix('.tmp')
            
            async with aiofiles.open(temp_file, "w") as f:
                await f.write(json.dumps(endpoints_data, indent=2))
            temp_file.replace(self.endpoints_file)
            
            logging.info("Configurations saved successfully")
            
        except Exception as e:
            logging.error(f"Configuration save error: {e}")
            raise

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        uptime = time.time() - self.start_time
        
        return {
            "status": "healthy" if self._initialized else "initializing",
            "uptime_seconds": uptime,
            "requests_total": self.request_count,
            "errors_total": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "services_count": len(self.services),
            "endpoints_count": len(self.endpoints),
            "cache_stats": await self.cache.get_stats(),
            "memory_usage_mb": self._get_memory_usage(),
            "circuit_breaker_stats": self._get_circuit_breaker_stats()
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        stats = {}
        for service, cb_stats in self.circuit_breaker.services.items():
            stats[service] = {
                "state": cb_stats.state.value,
                "failure_count": cb_stats.failure_count,
                "success_count": cb_stats.success_count
            }
        return stats

    async def close(self):
        """Production-grade shutdown procedure"""
        logging.info("Starting gateway shutdown...")
        errors = []
        
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
            # Close HTTP session
            if self.session:
                try:
                    await self.session.close()
                    # Wait for connections to close
                    await asyncio.sleep(0.1)
                except Exception as e:
                    errors.append(f"Session close error: {e}")
                    
            # Close components
            components = [
                ("cache", self.cache.close()),
                ("rate_limiter", self.rate_limiter.close())
            ]
            
            for name, close_coro in components:
                try:
                    await close_coro
                except Exception as e:
                    errors.append(f"{name} close error: {e}")
                    
            # Cancel monitoring tasks
            tasks = [self._health_task, self._config_watcher_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()
                    
            # Wait for tasks to complete
            if tasks:
                await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)
                
            self._initialized = False
            
            if errors:
                logging.warning(f"Shutdown completed with errors: {'; '.join(errors)}")
            else:
                logging.info("Gateway shutdown completed successfully")
                
        except Exception as e:
            logging.error(f"Critical error during shutdown: {e}")
            raise

# ===== FASTMCP SERVER IMPLEMENTATION =====

# Load production configuration
config = ProductionConfig.from_env()
setup_production_logging(config)

# Create FastMCP server with production settings
mcp = FastMCP(
    name="Universal API Gateway Production",
    dependencies=["aiohttp", "aiofiles", "aiosqlite", "pydantic", "psutil"]
)

# Global gateway instance
gateway = ProductionAPIGateway(config)

# ===== ENHANCED MCP TOOLS WITH PRODUCTION FEATURES =====

@mcp.tool(
    annotations={
        "title": "Configure API Service",
        "destructiveHint": False,
        "openWorldHint": False
    }
)
async def configure_api_service(
    name: str,
    base_url: str, 
    auth_type: str,
    auth_config: Optional[Dict[str, Any]] = None,
    default_headers: Optional[Dict[str, str]] = None,
    rate_limit: Optional[int] = None,
    timeout: int = 30,
    retry_attempts: int = 3,
    cache_ttl: int = 300,
    circuit_breaker_enabled: bool = True,
    health_check_path: Optional[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """Configure a new API service with production features"""
    try:
        await gateway.initialize()
        
        service_config = APIServiceConfig(
            name=name,
            base_url=base_url,
            auth_type=auth_type,
            auth_config=AuthConfig(**auth_config) if auth_config else None,
            default_headers=default_headers or {},
            rate_limit=rate_limit,
            timeout=timeout,
            retry_attempts=retry_attempts,
            cache_ttl=cache_ttl,
            circuit_breaker_enabled=circuit_breaker_enabled,
            health_check_path=health_check_path
        )
        
        gateway.services[name] = service_config
        await gateway.save_configurations()
        
        if ctx:
            await ctx.info(f"Successfully configured service: {name}")
        
        return {
            "status": "success", 
            "message": f"✅ Configured API service: {name}",
            "service_count": len(gateway.services),
            "config": service_config.dict()
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to configure service {name}: {str(e)}")
        return {"status": "error", "error": str(e)}

@mcp.tool(
    annotations={
        "title": "Configure API Endpoint",
        "destructiveHint": False,
        "openWorldHint": False
    }
)
async def configure_api_endpoint(
    name: str,
    service: str,
    path: str,
    method: str,
    description: str,
    parameters_schema: Optional[Dict[str, Any]] = None,
    response_schema: Optional[Dict[str, Any]] = None,
    cache_enabled: bool = True,
    requires_auth: bool = True,
    tags: Optional[List[str]] = None,
    max_request_size: Optional[int] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """Configure a new API endpoint with validation"""
    try:
        await gateway.initialize()
        
        if service not in gateway.services:
            raise ValueError(f"Service '{service}' not found. Configure it first.")
        
        endpoint_config = EndpointConfig(
            name=name,
            service=service,
            path=path,
            method=method.upper(),
            description=description,
            parameters_schema=parameters_schema or {},
            response_schema=response_schema,
            cache_enabled=cache_enabled,
            requires_auth=requires_auth,
            tags=tags or [],
            max_request_size=max_request_size
        )
        
        gateway.endpoints[name] = endpoint_config
        await gateway.save_configurations()
        
        if ctx:
            await ctx.info(f"Successfully configured endpoint: {name}")
        
        return {
            "status": "success",
            "message": f"✅ Configured API endpoint: {name}",
            "endpoint_count": len(gateway.endpoints),
            "config": endpoint_config.dict()
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to configure endpoint {name}: {str(e)}")
        return {"status": "error", "error": str(e)}

@mcp.tool(
    annotations={
        "title": "Make API Request",
        "destructiveHint": False,
        "openWorldHint": True
    }
)
async def api_request(
    service: str,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    override_method: Optional[str] = None,
    client_id: Optional[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """Make a production-grade API request with full monitoring"""
    try:
        await gateway.initialize()
        
        if ctx:
            await ctx.report_progress(progress=0, total=100, message="Starting request")
        
        result = await gateway.make_api_request(
            service_name=service,
            endpoint_name=endpoint,
            params=params,
            body=body,
            extra_headers=headers,
            override_method=override_method,
            client_id=client_id,
            ctx=ctx
        )
        
        if ctx:
            await ctx.report_progress(progress=100, total=100, message="Request completed")
        
        return {
            "status": "success",
            "response": result
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"API request failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "service": service,
            "endpoint": endpoint
        }

@mcp.tool(
    annotations={
        "title": "Batch API Requests",
        "destructiveHint": False,
        "openWorldHint": True
    }
)
async def batch_api_requests(
    requests: List[Dict[str, Any]],
    max_concurrent: int = 10,
    client_id: Optional[str] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """Execute multiple API requests with production-grade concurrency control"""
    try:
        await gateway.initialize()
        
        if ctx:
            await ctx.info(f"Starting batch execution of {len(requests)} requests")
            await ctx.report_progress(progress=0, total=len(requests))
        
        # Limit concurrency for production stability
        max_concurrent = min(max_concurrent, config.max_connections_per_host // 2)
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0
        
        async def execute_single_request(req_data, req_index):
            nonlocal completed
            async with semaphore:
                try:
                    result = await gateway.make_api_request(
                        service_name=req_data["service"],
                        endpoint_name=req_data["endpoint"],
                        params=req_data.get("params"),
                        body=req_data.get("body"),
                        extra_headers=req_data.get("headers"),
                        override_method=req_data.get("override_method"),
                        client_id=client_id,
                        ctx=None  # Don't pass ctx to avoid spam
                    )
                    
                    completed += 1
                    if ctx and completed % 5 == 0:  # Update every 5 requests
                        await ctx.report_progress(progress=completed, total=len(requests))
                    
                    return {
                        "index": req_index, 
                        "status": "success", 
                        "result": result
                    }
                    
                except Exception as e:
                    completed += 1
                    return {
                        "index": req_index, 
                        "status": "error", 
                        "error": str(e),
                        "request": req_data
                    }
        
        # Validate and execute requests
        tasks = []
        for i, req in enumerate(requests):
            if "service" not in req or "endpoint" not in req:
                logging.warning(f"Invalid request {i}: missing service or endpoint")
                continue
            tasks.append(execute_single_request(req, i))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        failed = len(results) - successful
        
        if ctx:
            await ctx.report_progress(progress=len(requests), total=len(requests))
            await ctx.info(f"Batch execution completed: {successful} successful, {failed} failed")
        
        return {
            "status": "completed",
            "total_requests": len(requests),
            "processed_requests": len(tasks),
            "successful": successful,
            "failed": failed,
            "results": results
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Batch execution failed: {str(e)}")
        return {"status": "error", "error": str(e)}

@mcp.tool(
    annotations={
        "title": "Get Gateway Health Status",
        "readOnlyHint": True,
        "openWorldHint": False
    }
)
async def get_health_status(ctx: Context = None) -> Dict[str, Any]:
    """Get comprehensive gateway health and performance metrics"""
    try:
        await gateway.initialize()
        health_data = await gateway.get_health_status()
        
        return {
            "status": "success",
            "health": health_data
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to get health status: {str(e)}")
        return {"status": "error", "error": str(e)}

@mcp.tool(
    annotations={
        "title": "List Services",
        "readOnlyHint": True,
        "openWorldHint": False
    }
)
async def list_services(include_stats: bool = True, ctx: Context = None) -> Dict[str, Any]:
    """List all configured services with optional statistics"""
    try:
        await gateway.initialize()
        
        services = []
        for name, config in gateway.services.items():
            service_info = {
                "name": name,
                "base_url": config.base_url,
                "auth_type": config.auth_type,
                "rate_limit": config.rate_limit,
                "timeout": config.timeout,
                "cache_ttl": config.cache_ttl,
                "circuit_breaker_enabled": config.circuit_breaker_enabled,
                "endpoint_count": len([e for e in gateway.endpoints.values() if e.service == name])
            }
            
            if include_stats:
                # Add circuit breaker stats if available
                cb_stats = gateway._get_circuit_breaker_stats().get(name)
                if cb_stats:
                    service_info["circuit_breaker_state"] = cb_stats["state"]
                    service_info["failure_count"] = cb_stats["failure_count"]
                    
            services.append(service_info)
        
        return {
            "status": "success",
            "total_services": len(services),
            "services": services
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to list services: {str(e)}")
        return {"status": "error", "error": str(e)}

@mcp.tool(
    annotations={
        "title": "Clear Cache",
        "destructiveHint": True,
        "openWorldHint": False
    }
)
async def clear_cache(ctx: Context = None) -> Dict[str, Any]:
    """Clear all cached responses with confirmation"""
    try:
        await gateway.initialize()
        
        # Get stats before clearing
        stats_before = await gateway.cache.get_stats()
        
        # Clear cache
        async with gateway.cache.lock:
            async with aiosqlite.connect(str(gateway.cache.db_path)) as db:
                result = await db.execute("DELETE FROM cache")
                await db.commit()
                entries_removed = result.rowcount
        
        if ctx:
            await ctx.info(f"Cache cleared: {entries_removed} entries removed")
        
        return {
            "status": "success",
            "message": f"✅ Cache cleared successfully",
            "entries_removed": entries_removed,
            "stats_before": stats_before
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to clear cache: {str(e)}")
        return {"status": "error", "error": str(e)}

# ===== GRACEFUL SHUTDOWN HANDLING =====

async def graceful_shutdown():
    """Handle graceful shutdown with proper cleanup"""
    logging.info("Received shutdown signal, initiating graceful shutdown...")
    
    try:
        await gateway.close()
        logging.info("Gateway shutdown completed")
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")
    finally:
        # Cancel any remaining tasks
        tasks = [task for task in asyncio.all_tasks() if not task.done()]
        for task in tasks:
            if task != asyncio.current_task():
                task.cancel()
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

# Signal handlers for graceful shutdown
def signal_handler(sig_num, frame):
    """Handle shutdown signals"""
    logging.info(f"Received signal {sig_num}")
    asyncio.create_task(graceful_shutdown())

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ===== MAIN ENTRY POINT =====

if __name__ == "__main__":
    try:
        logging.info("Starting Universal API Gateway Production Server...")
        mcp.run(
            transport=config.transport.value,
            host=config.host if config.transport == TransportType.HTTP else None,
            port=config.port if config.transport == TransportType.HTTP else None,
            log_level=config.log_level.value.lower()
        )
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt")
    except Exception as e:
        logging.error(f"Server startup failed: {e}")
        sys.exit(1)
    finally:
        logging.info("Server shutdown complete")