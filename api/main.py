"""
Story Memory System - FastAPI Application

Main entry point for the AI interactive storytelling platform with persistent memory.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("Starting Story Memory System...")
    print(f"Debug mode: {os.getenv('DEBUG', 'False')}")

    yield

    # Shutdown
    print("Shutting down Story Memory System...")


# Create FastAPI app
app = FastAPI(
    title="Story Memory System",
    description="AI interactive storytelling platform with persistent memory",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Story Memory System",
        "version": "0.1.0",
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "database": "not_configured",  # Will be updated when DB is connected
        "extraction_service": "not_configured",
    }


# Import and include routers (will be uncommented as routes are implemented)
# from api.routes import stories, sessions, memory
# app.include_router(stories.router)
# app.include_router(sessions.router)
# app.include_router(memory.router)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() == "true"

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=debug,
    )
