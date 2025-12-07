"""
Supabase database client.
"""

from typing import Optional
from supabase import create_client, Client
from config import require_secret


class DatabaseClient:
    """Wrapper for Supabase client with convenience methods."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize database client.

        Args:
            url: Supabase URL (defaults to SUPABASE_URL from secret manager)
            key: Supabase key (defaults to SUPABASE_KEY from secret manager)
        """
        # Get credentials from secret manager if not provided
        self.url = url or require_secret("SUPABASE_URL")
        self.key = key or require_secret("SUPABASE_KEY")

        self.client: Client = create_client(self.url, self.key)

    def get_client(self) -> Client:
        """Get the underlying Supabase client."""
        return self.client

    # Add convenience methods here as needed
    # For example:
    # def get_story(self, story_id: str):
    #     return self.client.table("stories").select("*").eq("id", story_id).execute()
