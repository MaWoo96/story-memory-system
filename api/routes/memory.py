"""
Memory-related API endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from uuid import UUID
from schemas.api import (
    SessionCompleteRequest,
    SessionCompleteResponse,
    ContextResponse,
    GameStateResponse,
)


router = APIRouter(prefix="/api", tags=["memory"])


@router.post("/sessions/{story_id}/complete", response_model=SessionCompleteResponse)
async def complete_session(
    story_id: UUID,
    request: SessionCompleteRequest,
) -> SessionCompleteResponse:
    """
    Process a completed session: extract memories and store them.

    Steps:
    1. Create session record
    2. Get existing entities for context
    3. Run extraction
    4. Store results
    5. Update story summary
    """
    # TODO: Implement session completion
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/sessions/{story_id}/context", response_model=ContextResponse)
async def get_session_context(story_id: UUID) -> ContextResponse:
    """
    Get context for starting a new session.

    Returns:
    - Story summary
    - Recent session summaries
    - Important entities
    - Current game state
    - Pending decisions
    """
    # TODO: Implement context retrieval
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/stories/{story_id}/search")
async def search_story_memories(
    story_id: UUID,
    q: str,
    limit: int = 10,
):
    """
    Search story memories for specific facts.
    """
    # TODO: Implement memory search
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/stories/{story_id}/summary")
async def get_story_summary(story_id: UUID):
    """
    Get full story summary and stats.
    """
    # TODO: Implement story summary
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/stories/{story_id}/state", response_model=GameStateResponse)
async def get_game_state(story_id: UUID) -> GameStateResponse:
    """
    Get current protagonist stats, skills, inventory, and NPC states.
    """
    # TODO: Implement game state retrieval
    raise HTTPException(status_code=501, detail="Not implemented yet")
