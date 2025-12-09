"""
Story Memory System - FastAPI Application

Main entry point for the AI interactive storytelling platform with persistent memory.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables (override existing to use project-specific config)
load_dotenv(override=True)


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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

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
from api.routes.images import router as images_router
from api.routes.frontend import router as frontend_router

# Frontend router first so it takes precedence for /sessions/{id}/context
app.include_router(frontend_router)
app.include_router(memory_router)
app.include_router(images_router)
from api.routes.stories import router as stories_router
app.include_router(stories_router)


# ============================================
# IMAGE GENERATOR UI
# ============================================

@app.get("/generator")
async def image_generator_ui():
    """Serve the image generator UI."""
    html_path = STATIC_DIR / "image-generator.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return {"error": "UI not found", "path": str(html_path)}


# Static file mounts (MUST be after route definitions)
app.mount("/images", StaticFiles(directory="/tmp/story-images"), name="images")

# Serve generated images from local directory
GENERATED_IMAGES_DIR = PROJECT_ROOT / "generated_images"
if GENERATED_IMAGES_DIR.exists():
    app.mount("/generated_images", StaticFiles(directory=str(GENERATED_IMAGES_DIR)), name="generated_images")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
