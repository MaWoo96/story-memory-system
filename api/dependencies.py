"""
Dependency injection for FastAPI routes.

Provides shared dependencies like database clients, service instances, etc.
"""

from functools import lru_cache
import os
from typing import Optional


@lru_cache()
def get_supabase_client():
    """
    Get Supabase client instance (cached).

    Returns:
        Supabase client instance

    Raises:
        ValueError: If Supabase credentials are not configured
    """
    from supabase import create_client, Client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")

    return create_client(url, key)


@lru_cache()
def get_extraction_service():
    """
    Get extraction service instance (cached).

    Returns:
        ExtractionService instance

    Raises:
        ValueError: If XAI_API_KEY is not configured
    """
    api_key = os.getenv("XAI_API_KEY")

    if not api_key:
        raise ValueError("XAI_API_KEY must be set in environment")

    # Will be implemented when extraction service is created
    # from services.extraction import ExtractionService
    # return ExtractionService()

    return None


def get_db():
    """
    Get database connection (for dependency injection).

    Yields:
        Database connection
    """
    client = get_supabase_client()
    try:
        yield client
    finally:
        # Cleanup if needed
        pass


def get_current_user(
    # Add authentication dependency when implemented
    # token: str = Depends(oauth2_scheme)
):
    """
    Get current authenticated user.

    This is a placeholder for future authentication.

    Returns:
        User information
    """
    # TODO: Implement authentication
    return {"user_id": "placeholder"}
