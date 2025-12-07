"""
Supabase database client.
"""

import os
from typing import Optional
from supabase import create_client, Client


class DatabaseClient:
    """Wrapper for Supabase client with convenience methods."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize database client.

        Args:
            url: Supabase URL (defaults to SUPABASE_URL env var)
            key: Supabase key (defaults to SUPABASE_KEY env var)
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")

        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

        self.client: Client = create_client(self.url, self.key)

    def get_client(self) -> Client:
        """Get the underlying Supabase client."""
        return self.client

    # Add convenience methods here as needed
    # For example:
    # def get_story(self, story_id: str):
    #     return self.client.table("stories").select("*").eq("id", story_id).execute()
