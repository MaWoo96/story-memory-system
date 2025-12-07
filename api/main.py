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

    # Verify required environment variables
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "XAI_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"Warning: Missing environment variables: {', '.join(missing)}")
    else:
        print("All required environment variables configured")

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
    db_status = "not_configured"
    extraction_status = "not_configured"

    try:
        from api.dependencies import get_supabase_client
        client = get_supabase_client()
        # Try a simple query to verify connection
        client.table("stories").select("id").limit(1).execute()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"

    try:
        from api.dependencies import get_extraction_service
        service = get_extraction_service()
        if service:
            extraction_status = "configured"
    except Exception as e:
        extraction_status = f"error: {str(e)[:50]}"

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "extraction_service": extraction_status,
    }


# Import and include routers
from api.routes.memory import router as memory_router
app.include_router(memory_router)


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
