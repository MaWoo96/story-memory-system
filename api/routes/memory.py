"""
Memory-related API endpoints.

Handles session completion, context retrieval, memory search, and game state.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from uuid import UUID
from typing import Optional
from datetime import datetime

from api.dependencies import (
    get_supabase_client,
    get_extraction_service,
    get_storage_service,
    get_context_service,
)
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
    background_tasks: BackgroundTasks,
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
    db = get_supabase_client()
    extraction_service = get_extraction_service()
    storage_service = get_storage_service()

    # 1. Verify story exists and get info
    story_result = db.table("stories").select(
        "id, premise, story_summary"
    ).eq("id", str(story_id)).execute()

    if not story_result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    story = story_result.data[0]

    # 2. Get current session number
    sessions_result = db.table("sessions").select(
        "session_number"
    ).eq("story_id", str(story_id)).order(
        "session_number", desc=True
    ).limit(1).execute()

    next_session_number = 1
    previous_summary = None

    if sessions_result.data:
        next_session_number = sessions_result.data[0]["session_number"] + 1
        # Get previous session summary for context
        prev_session = db.table("sessions").select("summary").eq(
            "story_id", str(story_id)
        ).eq("session_number", sessions_result.data[0]["session_number"]).execute()
        if prev_session.data and prev_session.data[0].get("summary"):
            previous_summary = prev_session.data[0]["summary"]

    # 3. Create session record
    session_result = db.table("sessions").insert({
        "story_id": str(story_id),
        "session_number": next_session_number,
        "transcript": request.transcript,
        "status": "processing",
    }).execute()

    session_id = UUID(session_result.data[0]["id"])

    try:
        # 4. Get existing entities for extraction context
        existing_entities = await storage_service.get_existing_entities(story_id)

        # 5. Run extraction
        extraction = extraction_service.extract_session(
            transcript=request.transcript,
            story_premise=story.get("premise"),
            existing_entities=existing_entities,
            previous_summary=previous_summary,
        )

        # 6. Store extraction results
        stats = await storage_service.store_extraction(
            story_id=story_id,
            session_id=session_id,
            extraction=extraction,
        )

        # 7. Update session status
        db.table("sessions").update({
            "status": "processed",
            "processed_at": datetime.utcnow().isoformat(),
            "ended_at": datetime.utcnow().isoformat(),
        }).eq("id", str(session_id)).execute()

        return SessionCompleteResponse(
            session_id=session_id,
            session_number=next_session_number,
            summary=extraction.session_summary,
            stats={
                "entities_created": stats["entities_created"],
                "entities_updated": stats["entities_updated"],
                "events_recorded": stats["events_recorded"],
                "decisions_captured": stats["decisions_captured"],
                "facts_created": stats["facts_created"],
                "relationships_created": stats["relationships_created"],
            },
        )

    except Exception as e:
        # Mark session as failed
        db.table("sessions").update({
            "status": "failed",
            "ended_at": datetime.utcnow().isoformat(),
        }).eq("id", str(session_id)).execute()

        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


@router.get("/sessions/{story_id}/context")
async def get_session_context(
    story_id: UUID,
    max_entities: int = Query(default=20, ge=1, le=50),
    recent_sessions: int = Query(default=3, ge=1, le=10),
):
    """
    Get context for starting a new session.

    Returns:
    - Story summary
    - Recent session summaries
    - Important entities with facts and relationships
    - Current game state
    - Pending decisions
    """
    context_service = get_context_service()

    try:
        context = await context_service.build_session_context(
            story_id=story_id,
            max_entities=max_entities,
            recent_sessions=recent_sessions,
        )

        return {
            "system_context": context["system_context"],
            "story_summary": context["story_summary"],
            "recent_sessions": context["recent_sessions"],
            "entity_reference": context["entity_reference"],
            "entities": context["entities"],
            "protagonist_state": context["protagonist_state"],
            "protagonist_data": context["protagonist_data"],
            "relationship_meters": context["relationship_meters"],
            "character_states": context["character_states"],
            "pending_decisions": context["pending_decisions"],
            "pending_decisions_data": context["pending_decisions_data"],
            "world_state": context["world_state"],
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build context: {str(e)}")


@router.get("/stories/{story_id}/search")
async def search_story_memories(
    story_id: UUID,
    q: str = Query(description="Search query"),
    limit: int = Query(default=10, ge=1, le=100),
    entity_types: Optional[str] = Query(
        default=None,
        description="Comma-separated entity types to filter"
    ),
    include_facts: bool = Query(default=True),
    include_events: bool = Query(default=True),
):
    """
    Search story memories for specific facts.

    Searches across entities, facts, and events.
    """
    context_service = get_context_service()

    # Parse entity types
    types_list = None
    if entity_types:
        types_list = [t.strip() for t in entity_types.split(",")]

    try:
        results = await context_service.search_memories(
            story_id=story_id,
            query=q,
            limit=limit,
            entity_types=types_list,
            include_facts=include_facts,
            include_events=include_events,
        )

        return {"query": q, "count": len(results), "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/stories/{story_id}/summary")
async def get_story_summary(story_id: UUID):
    """
    Get full story summary and stats.
    """
    db = get_supabase_client()

    # Get story
    story_result = db.table("stories").select("*").eq(
        "id", str(story_id)
    ).execute()

    if not story_result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    story = story_result.data[0]

    # Get session count
    sessions = db.table("sessions").select(
        "id", count="exact"
    ).eq("story_id", str(story_id)).eq("status", "processed").execute()

    # Get entity count
    entities = db.table("entities").select(
        "id", count="exact"
    ).eq("story_id", str(story_id)).neq("status", "inactive").execute()

    # Get event count
    events = db.table("events").select(
        "id", count="exact"
    ).eq("story_id", str(story_id)).execute()

    # Get arc summaries
    arcs = db.table("arc_summaries").select("*").eq(
        "story_id", str(story_id)
    ).order("arc_number").execute()

    return {
        "story": story,
        "stats": {
            "sessions_completed": sessions.count if sessions.count else 0,
            "entities_tracked": entities.count if entities.count else 0,
            "events_recorded": events.count if events.count else 0,
        },
        "arcs": arcs.data if arcs.data else [],
    }


@router.get("/stories/{story_id}/state", response_model=GameStateResponse)
async def get_game_state(story_id: UUID) -> GameStateResponse:
    """
    Get current protagonist stats, skills, inventory, and NPC states.
    """
    db = get_supabase_client()

    # Verify story exists
    story_result = db.table("stories").select("id").eq(
        "id", str(story_id)
    ).execute()

    if not story_result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    # Get stats
    stats = db.table("protagonist_stats").select(
        "stat_name, current_value, max_value, stat_type"
    ).eq("story_id", str(story_id)).execute()

    # Get skills
    skills = db.table("protagonist_skills").select(
        "skill_name, rank, description, mechanical_effect, requirements, cooldown"
    ).eq("story_id", str(story_id)).execute()

    # Get active inventory
    inventory = db.table("protagonist_inventory").select(
        "item_name, description, properties, equipped"
    ).eq("story_id", str(story_id)).is_(
        "lost_session_id", "null"
    ).execute()

    # Get active status effects
    effects = db.table("protagonist_status_effects").select(
        "effect_name, description, is_temporary"
    ).eq("story_id", str(story_id)).is_(
        "removed_session_id", "null"
    ).execute()

    # Get character states
    char_states_result = db.table("character_states").select(
        "entity_id, stat_type, current_value, max_value, label"
    ).eq("story_id", str(story_id)).execute()

    # Resolve entity names
    character_states = []
    for state in char_states_result.data:
        entity = db.table("entities").select("canonical_name").eq(
            "id", state["entity_id"]
        ).execute()
        if entity.data:
            character_states.append({
                "character": entity.data[0]["canonical_name"],
                "stat_type": state["stat_type"],
                "value": state["current_value"],
                "max": state["max_value"],
                "label": state["label"],
            })

    return GameStateResponse(
        protagonist_stats=stats.data or [],
        protagonist_skills=skills.data or [],
        inventory=inventory.data or [],
        status_effects=effects.data or [],
        character_states=character_states,
    )


@router.get("/stories/{story_id}/timeline")
async def get_story_timeline(
    story_id: UUID,
    start_session: Optional[int] = Query(default=None),
    end_session: Optional[int] = Query(default=None),
    event_types: Optional[str] = Query(
        default=None,
        description="Comma-separated event types to filter"
    ),
    limit: int = Query(default=100, ge=1, le=500),
):
    """
    Get chronological timeline of events, optionally filtered.
    """
    context_service = get_context_service()

    # Parse event types
    types_list = None
    if event_types:
        types_list = [t.strip() for t in event_types.split(",")]

    try:
        timeline = await context_service.get_story_timeline(
            story_id=story_id,
            start_session=start_session,
            end_session=end_session,
            event_types=types_list,
            limit=limit,
        )

        return {"count": len(timeline), "events": timeline}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get timeline: {str(e)}")


@router.get("/entities/{entity_id}")
async def get_entity(entity_id: UUID):
    """
    Get full entity details including all facts, relationships, and history.
    """
    context_service = get_context_service()

    try:
        entity_detail = await context_service.get_entity_detail(entity_id)

        if not entity_detail:
            raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

        return entity_detail

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get entity: {str(e)}")


@router.post("/entities/merge")
async def merge_entities(
    primary_entity_id: UUID,
    secondary_entity_id: UUID,
    reason: str,
):
    """
    Merge two entities when they're discovered to be the same.

    Moves all facts, relationships, and event participations from
    secondary to primary, then soft-deletes secondary.
    """
    db = get_supabase_client()
    storage_service = get_storage_service()

    # Get both entities
    primary = db.table("entities").select("story_id").eq(
        "id", str(primary_entity_id)
    ).execute()

    secondary = db.table("entities").select("story_id").eq(
        "id", str(secondary_entity_id)
    ).execute()

    if not primary.data:
        raise HTTPException(status_code=404, detail="Primary entity not found")

    if not secondary.data:
        raise HTTPException(status_code=404, detail="Secondary entity not found")

    if primary.data[0]["story_id"] != secondary.data[0]["story_id"]:
        raise HTTPException(
            status_code=400,
            detail="Cannot merge entities from different stories"
        )

    story_id = UUID(primary.data[0]["story_id"])

    try:
        await storage_service.merge_entities(
            story_id=story_id,
            primary_entity_id=primary_entity_id,
            secondary_entity_id=secondary_entity_id,
            reason=reason,
            merged_by="user",
        )

        return {
            "success": True,
            "primary_entity_id": str(primary_entity_id),
            "merged_entity_id": str(secondary_entity_id),
            "reason": reason,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Merge failed: {str(e)}")


# ============================================
# STORY CRUD ENDPOINTS
# ============================================

@router.post("/stories")
async def create_story(
    title: str,
    premise: Optional[str] = None,
    user_id: str = "placeholder",  # Will come from auth
):
    """Create a new story."""
    db = get_supabase_client()

    result = db.table("stories").insert({
        "user_id": user_id,
        "title": title,
        "premise": premise,
        "status": "active",
    }).execute()

    return result.data[0]


@router.get("/stories")
async def list_stories(
    user_id: str = "placeholder",  # Will come from auth
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
):
    """List stories for a user."""
    db = get_supabase_client()

    query = db.table("stories").select("*").eq("user_id", user_id)

    if status:
        query = query.eq("status", status)

    result = query.order("updated_at", desc=True).limit(limit).execute()

    return {"stories": result.data}


@router.get("/stories/{story_id}")
async def get_story(story_id: UUID):
    """Get a story by ID."""
    db = get_supabase_client()

    result = db.table("stories").select("*").eq("id", str(story_id)).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    return result.data[0]


@router.patch("/stories/{story_id}")
async def update_story(
    story_id: UUID,
    title: Optional[str] = None,
    premise: Optional[str] = None,
    status: Optional[str] = None,
):
    """Update a story."""
    db = get_supabase_client()

    update_data = {"updated_at": datetime.utcnow().isoformat()}

    if title is not None:
        update_data["title"] = title
    if premise is not None:
        update_data["premise"] = premise
    if status is not None:
        update_data["status"] = status

    result = db.table("stories").update(update_data).eq(
        "id", str(story_id)
    ).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    return result.data[0]
