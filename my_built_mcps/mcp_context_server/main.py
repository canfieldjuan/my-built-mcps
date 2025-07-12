"""
main.py

Main FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mcp_context_server.config.settings import settings
from mcp_context_server.core.session_manager import session_manager
from mcp_context_server.api.v1 import routes as v1_routes
from mcp_context_server.api.v1 import websocket as ws_routes
from mcp_context_server.api.mcp import handlers as mcp_handlers

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MCP Context Server...")
    
    # Validate settings
    settings.validate()
    
    # Start session manager
    await session_manager.start()
    
    logger.info("MCP Context Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP Context Server...")
    
    # Stop session manager
    await session_manager.stop()
    
    logger.info("MCP Context Server shut down")


# Create FastAPI app
app = FastAPI(
    title="MCP Context Graph Server",
    version="2.0.0",
    description="Advanced memory management system for AI conversations",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(v1_routes.router)
app.include_router(ws_routes.router)
app.include_router(mcp_handlers.router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MCP Context Graph Server",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "docs": "/docs",
            "redoc": "/redoc",
            "mcp": "/mcp"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "mcp_context_server.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )