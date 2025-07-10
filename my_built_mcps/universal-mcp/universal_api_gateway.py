#!/usr/bin/env python3
"""
Universal API Gateway MCP Server
One MCP server to connect to any API without building individual servers.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import aiohttp
import aiofiles
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
import mcp.types as types
from urllib.parse import urljoin, urlencode
import hashlib
import sqlite3
from datetime import datetime, timedelta

@dataclass
class APIConfig:
    """Configuration for an API service"""
    name: str
    base_url: str
    auth_type: str  # "none", "api_key", "bearer", "oauth", "basic"
    auth_config: Dict[str, Any]
    default_headers: Dict[str, str]
    rate_limit: Optional[int] = None  # requests per minute
    timeout: int = 30
    retry_attempts: int = 3
    cache_ttl: int = 300  # 5 minutes default

@dataclass
class APIEndpoint:
    """Configuration for a specific API endpoint"""
    service: str
    path: str
    method: str
    description: str
    parameters: Dict[str, Any]
    response_schema: Optional[Dict[str, Any]] = None
    cache_enabled: bool = True
    requires_auth: bool = True

class CacheManager:
    """Simple SQLite-based cache manager"""
    
    def __init__(self, db_path: str = "api_cache.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                expires_at REAL
            )
        """)
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
            (key, time.time())
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def set(self, key: str, value: str, ttl: int):
        expires_at = time.time() + ttl
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, value, expires_at)
        )
        conn.commit()
        conn.close()
    
    def cleanup_expired(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM cache WHERE expires_at <= ?", (time.time(),))
        conn.commit()
        conn.close()

class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self):
        self.requests = {}
    
    def can_request(self, service: str, limit: int) -> bool:
        now = time.time()
        minute_ago = now - 60
        
        if service not in self.requests:
            self.requests[service] = []
        
        # Clean old requests
        self.requests[service] = [req_time for req_time in self.requests[service] if req_time > minute_ago]
        
        # Check if we can make another request
        if len(self.requests[service]) < limit:
            self.requests[service].append(now)
            return True
        
        return False

class UniversalAPIGateway:
    """Universal API Gateway for MCP"""
    
    def __init__(self):
        self.services: Dict[str, APIConfig] = {}
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.cache = CacheManager()
        self.rate_limiter = RateLimiter()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Load configurations
        self._load_configs()
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _load_configs(self):
        """Load API configurations from files"""
        try:
            # Load services
            if Path("api_services.json").exists():
                with open("api_services.json", "r") as f:
                    services_data = json.load(f)
                    for name, config in services_data.items():
                        self.services[name] = APIConfig(**config)
            
            # Load endpoints
            if Path("api_endpoints.json").exists():
                with open("api_endpoints.json", "r") as f:
                    endpoints_data = json.load(f)
                    for name, config in endpoints_data.items():
                        self.endpoints[name] = APIEndpoint(**config)
        except Exception as e:
            print(f"Error loading configs: {e}")
    
    async def save_configs(self):
        """Save configurations to files"""
        try:
            # Save services
            services_data = {name: asdict(config) for name, config in self.services.items()}
            async with aiofiles.open("api_services.json", "w") as f:
                await f.write(json.dumps(services_data, indent=2))
            
            # Save endpoints
            endpoints_data = {name: asdict(config) for name, config in self.endpoints.items()}
            async with aiofiles.open("api_endpoints.json", "w") as f:
                await f.write(json.dumps(endpoints_data, indent=2))
        except Exception as e:
            print(f"Error saving configs: {e}")
    
    def _get_cache_key(self, service: str, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        key_data = f"{service}:{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _prepare_auth(self, service_config: APIConfig) -> Dict[str, str]:
        """Prepare authentication headers"""
        headers = {}
        auth_type = service_config.auth_type
        auth_config = service_config.auth_config
        
        if auth_type == "api_key":
            if auth_config.get("location") == "header":
                headers[auth_config["key"]] = auth_config["value"]
        elif auth_type == "bearer":
            headers["Authorization"] = f"Bearer {auth_config['token']}"
        elif auth_type == "basic":
            import base64
            creds = f"{auth_config['username']}:{auth_config['password']}"
            encoded = base64.b64encode(creds.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        
        return headers
    
    async def make_request(
        self,
        service_name: str,
        endpoint_name: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        override_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make API request"""
        
        # Get configurations
        if service_name not in self.services:
            raise ValueError(f"Service '{service_name}' not configured")
        
        if endpoint_name not in self.endpoints:
            raise ValueError(f"Endpoint '{endpoint_name}' not configured")
        
        service_config = self.services[service_name]
        endpoint_config = self.endpoints[endpoint_name]
        
        # Check rate limiting
        if service_config.rate_limit:
            if not self.rate_limiter.can_request(service_name, service_config.rate_limit):
                raise Exception(f"Rate limit exceeded for {service_name}")
        
        # Prepare request
        method = override_method or endpoint_config.method
        url = urljoin(service_config.base_url, endpoint_config.path)
        headers = {**service_config.default_headers}
        
        # Add authentication
        if endpoint_config.requires_auth:
            auth_headers = self._prepare_auth(service_config)
            headers.update(auth_headers)
        
        # Handle parameters
        if params:
            if method.upper() == "GET":
                url += "?" + urlencode(params)
            else:
                body = {**(body or {}), **params}
        
        # Check cache
        cache_key = None
        if endpoint_config.cache_enabled and method.upper() == "GET":
            cache_key = self._get_cache_key(service_name, endpoint_name, params or {})
            cached_response = self.cache.get(cache_key)
            if cached_response:
                return json.loads(cached_response)
        
        # Make request with retry logic
        for attempt in range(service_config.retry_attempts):
            try:
                async with self.session.request(
                    method,
                    url,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=service_config.timeout)
                ) as response:
                    
                    response_data = {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "data": None
                    }
                    
                    # Handle different content types
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        response_data["data"] = await response.json()
                    else:
                        response_data["data"] = await response.text()
                    
                    # Cache successful GET requests
                    if (cache_key and response.status == 200 and 
                        endpoint_config.cache_enabled):
                        self.cache.set(
                            cache_key, 
                            json.dumps(response_data), 
                            service_config.cache_ttl
                        )
                    
                    return response_data
                    
            except Exception as e:
                if attempt == service_config.retry_attempts - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("Max retry attempts exceeded")

# MCP Server Setup
app = Server("universal-api-gateway")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="configure_api_service",
            description="Configure a new API service",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Service name"},
                    "base_url": {"type": "string", "description": "Base URL of the API"},
                    "auth_type": {
                        "type": "string", 
                        "enum": ["none", "api_key", "bearer", "oauth", "basic"],
                        "description": "Authentication type"
                    },
                    "auth_config": {
                        "type": "object",
                        "description": "Authentication configuration"
                    },
                    "default_headers": {
                        "type": "object",
                        "description": "Default headers to send"
                    },
                    "rate_limit": {
                        "type": "integer",
                        "description": "Rate limit (requests per minute)"
                    }
                },
                "required": ["name", "base_url", "auth_type"]
            }
        ),
        types.Tool(
            name="configure_api_endpoint",
            description="Configure a new API endpoint",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Endpoint name"},
                    "service": {"type": "string", "description": "Service name"},
                    "path": {"type": "string", "description": "Endpoint path"},
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                        "description": "HTTP method"
                    },
                    "description": {"type": "string", "description": "Endpoint description"},
                    "parameters": {
                        "type": "object",
                        "description": "Parameter schema"
                    },
                    "cache_enabled": {
                        "type": "boolean",
                        "description": "Enable caching"
                    }
                },
                "required": ["name", "service", "path", "method", "description"]
            }
        ),
        types.Tool(
            name="api_request",
            description="Make an API request to any configured endpoint",
            inputSchema={
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "endpoint": {"type": "string", "description": "Endpoint name"},
                    "params": {
                        "type": "object",
                        "description": "Request parameters"
                    },
                    "body": {
                        "type": "object",
                        "description": "Request body (for POST/PUT)"
                    },
                    "method": {
                        "type": "string",
                        "description": "Override HTTP method"
                    }
                },
                "required": ["service", "endpoint"]
            }
        ),
        types.Tool(
            name="batch_api_requests",
            description="Make multiple API requests in parallel",
            inputSchema={
                "type": "object",
                "properties": {
                    "requests": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "service": {"type": "string"},
                                "endpoint": {"type": "string"},
                                "params": {"type": "object"},
                                "body": {"type": "object"}
                            },
                            "required": ["name", "service", "endpoint"]
                        }
                    }
                },
                "required": ["requests"]
            }
        ),
        types.Tool(
            name="list_services",
            description="List all configured API services",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="list_endpoints",
            description="List all configured API endpoints",
            inputSchema={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Filter by service name"
                    }
                }
            }
        ),
        types.Tool(
            name="test_endpoint",
            description="Test an API endpoint with sample data",
            inputSchema={
                "type": "object",
                "properties": {
                    "service": {"type": "string"},
                    "endpoint": {"type": "string"},
                    "sample_params": {"type": "object"}
                },
                "required": ["service", "endpoint"]
            }
        )
    ]

# Global gateway instance
gateway = None

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    global gateway
    
    if gateway is None:
        gateway = UniversalAPIGateway()
        await gateway.__aenter__()
    
    try:
        if name == "configure_api_service":
            config = APIConfig(
                name=arguments["name"],
                base_url=arguments["base_url"],
                auth_type=arguments["auth_type"],
                auth_config=arguments.get("auth_config", {}),
                default_headers=arguments.get("default_headers", {}),
                rate_limit=arguments.get("rate_limit"),
                timeout=arguments.get("timeout", 30),
                retry_attempts=arguments.get("retry_attempts", 3),
                cache_ttl=arguments.get("cache_ttl", 300)
            )
            gateway.services[config.name] = config
            await gateway.save_configs()
            return [types.TextContent(
                type="text",
                text=f"✅ Configured API service: {config.name}"
            )]
        
        elif name == "configure_api_endpoint":
            endpoint = APIEndpoint(
                service=arguments["service"],
                path=arguments["path"],
                method=arguments["method"],
                description=arguments["description"],
                parameters=arguments.get("parameters", {}),
                response_schema=arguments.get("response_schema"),
                cache_enabled=arguments.get("cache_enabled", True),
                requires_auth=arguments.get("requires_auth", True)
            )
            gateway.endpoints[arguments["name"]] = endpoint
            await gateway.save_configs()
            return [types.TextContent(
                type="text",
                text=f"✅ Configured API endpoint: {arguments['name']}"
            )]
        
        elif name == "api_request":
            result = await gateway.make_request(
                arguments["service"],
                arguments["endpoint"],
                arguments.get("params"),
                arguments.get("body"),
                arguments.get("method")
            )
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "batch_api_requests":
            tasks = []
            for req in arguments["requests"]:
                task = gateway.make_request(
                    req["service"],
                    req["endpoint"],
                    req.get("params"),
                    req.get("body")
                )
                tasks.append((req["name"], task))
            
            results = {}
            for name, task in tasks:
                try:
                    results[name] = await task
                except Exception as e:
                    results[name] = {"error": str(e)}
            
            return [types.TextContent(
                type="text",
                text=json.dumps(results, indent=2)
            )]
        
        elif name == "list_services":
            services = []
            for name, config in gateway.services.items():
                services.append({
                    "name": name,
                    "base_url": config.base_url,
                    "auth_type": config.auth_type,
                    "rate_limit": config.rate_limit
                })
            return [types.TextContent(
                type="text",
                text=json.dumps(services, indent=2)
            )]
        
        elif name == "list_endpoints":
            service_filter = arguments.get("service")
            endpoints = []
            for name, endpoint in gateway.endpoints.items():
                if not service_filter or endpoint.service == service_filter:
                    endpoints.append({
                        "name": name,
                        "service": endpoint.service,
                        "method": endpoint.method,
                        "path": endpoint.path,
                        "description": endpoint.description
                    })
            return [types.TextContent(
                type="text",
                text=json.dumps(endpoints, indent=2)
            )]
        
        elif name == "test_endpoint":
            # Test with provided or default params
            sample_params = arguments.get("sample_params", {})
            result = await gateway.make_request(
                arguments["service"],
                arguments["endpoint"],
                sample_params
            )
            return [types.TextContent(
                type="text",
                text=f"🧪 Test Result:\n{json.dumps(result, indent=2)}"
            )]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Error: {str(e)}"
        )]

async def main():
    # Import here to avoid issues with event loop
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="universal-api-gateway",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())