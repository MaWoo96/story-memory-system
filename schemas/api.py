"""
API request and response models.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


# ============================================
# REQUEST MODELS
# ============================================


class CreateStoryRequest(BaseModel):
    """Request to create a new story."""

    title: str = Field(description="Story title")
    premise: Optional[str] = Field(description="Story premise/setting", default=None)


class SessionCompleteRequest(BaseModel):
    """Request to mark a session as complete and extract memories."""

    transcript: str = Field(description="Full session transcript")


class SearchMemoriesRequest(BaseModel):
    """Request to search story memories."""

    query: str = Field(description="Search query")
    limit: int = Field(description="Maximum results", default=10, ge=1, le=100)


# ============================================
# RESPONSE MODELS
# ============================================


class StoryResponse(BaseModel):
    """Story information response."""

    id: UUID
    user_id: UUID
    title: str
    premise: Optional[str]
    status: str
    story_summary: Optional[str]
    current_situation: Optional[str]
    created_at: datetime
    updated_at: datetime


class SessionResponse(BaseModel):
    """Session information response."""

    id: UUID
    story_id: UUID
    session_number: int
    summary: Optional[str]
    key_moments: Optional[list[str]]
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    processed_at: Optional[datetime]


class SessionCompleteResponse(BaseModel):
    """Response after completing a session."""

    session_id: UUID
    session_number: int
    summary: str
    stats: dict = Field(description="Extracted stats summary")


class ContextResponse(BaseModel):
    """Context for starting a new session."""

    story_summary: Optional[str]
    recent_sessions: list[SessionResponse]
    important_entities: list[dict]
    current_state: dict
    pending_decisions: list[dict]


class GameStateResponse(BaseModel):
    """Current game state (stats, inventory, NPCs)."""

    protagonist_stats: list[dict]
    protagonist_skills: list[dict]
    inventory: list[dict]
    status_effects: list[dict]
    character_states: list[dict]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    database: str
    extraction_service: str
