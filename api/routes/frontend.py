"""
Frontend-compatible API endpoints.

These endpoints transform data to match the story-ui frontend's expected shapes.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from uuid import UUID
from typing import Optional
from datetime import datetime
import math

from api.dependencies import get_supabase_client
from schemas.api import (
    FrontendStory,
    FrontendStoryListResponse,
    FrontendCharacter,
    FrontendCharacterListResponse,
    FrontendCharacterDetailsResponse,
    FrontendGameState,
    FrontendGameStateResponse,
    FrontendProtagonist,
    FrontendStat,
    FrontendSkill,
    FrontendInventoryItem,
    FrontendNPCRelationship,
    FrontendWorldState,
    FrontendLocation,
    FrontendQuest,
    RelationshipMeter,
    CharacterRelationship,
    CharacterEvent,
    FrontendCompleteSessionResponse,
    GeneratePortraitRequest,
    GeneratePortraitResponse,
    UpdateStoryRequest,
    CreateStoryRequest,
    FrontendSemanticMemory,
    SemanticMemoryListResponse,
    UpdateMemoryRequest,
    CreateMemoryRequest,
    IntimacyHistoryResponse,
    IntimacyMetricsListResponse,
    PhysicalStateListResponse,
    FrontendSceneSummary,
    FrontendChapterSummary,
    FrontendArcSummary,
    SceneSummaryListResponse,
    ChapterSummaryListResponse,
    ArcSummaryListResponse,
    HierarchicalContextResponse,
)


router = APIRouter(prefix="/api", tags=["frontend"])


# ============================================
# HELPER FUNCTIONS
# ============================================


def importance_to_stars(importance: float) -> int:
    """Convert 0.0-1.0 importance to 1-5 stars."""
    return max(1, min(5, math.ceil(importance * 5)))


def entity_type_to_frontend(entity_type: str) -> str:
    """Map backend entity_type to frontend type."""
    if entity_type == "location":
        return "location"
    elif entity_type == "character":
        return "main"  # Default to main, could be refined based on importance
    else:
        return "side"


def stat_type_to_relationship_type(stat_type: str) -> str:
    """Map backend stat_type to frontend relationship type."""
    mapping = {
        "affection": "affection",
        "trust": "trust",
        "loyalty": "trust",
        "fear": "fear",
        "respect": "trust",
        "rivalry": "fear",
    }
    return mapping.get(stat_type, "trust")


def get_relationship_label(value: int, stat_type: str) -> str:
    """Generate a relationship label based on value."""
    if stat_type in ("affection", "trust", "loyalty"):
        if value >= 80:
            return "deeply devoted"
        elif value >= 60:
            return "warmly attached"
        elif value >= 40:
            return "friendly"
        elif value >= 20:
            return "cautiously respectful"
        else:
            return "distant"
    elif stat_type in ("fear", "rivalry"):
        if value >= 80:
            return "terrified"
        elif value >= 60:
            return "very wary"
        elif value >= 40:
            return "nervous"
        elif value >= 20:
            return "slightly uneasy"
        else:
            return "unafraid"
    return "neutral"


# ============================================
# STORIES
# ============================================


# Default user ID for development (single-user mode)
DEFAULT_USER_ID = "1be42d46-1c0c-45c6-9dd4-bd9647e9da32"


@router.get("/stories", response_model=FrontendStoryListResponse)
async def list_stories_frontend(
    user_id: str = Query(default=DEFAULT_USER_ID),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> FrontendStoryListResponse:
    """List stories in frontend-compatible format."""
    db = get_supabase_client()

    query = db.table("stories").select("id, title, created_at, updated_at").is_("deleted_at", "null")

    # Filter by user_id
    query = query.eq("user_id", user_id)

    if status:
        query = query.eq("status", status)

    result = query.order("updated_at", desc=True).limit(limit).execute()

    stories = [
        FrontendStory(
            id=str(s["id"]),
            title=s["title"],
            createdAt=s["created_at"],
            lastPlayed=s["updated_at"],
        )
        for s in result.data
    ]

    return FrontendStoryListResponse(stories=stories)


@router.post("/stories", response_model=FrontendStory)
async def create_story_frontend(
    request: CreateStoryRequest,
    user_id: Optional[str] = Query(default=None),
) -> FrontendStory:
    """Create a story and return in frontend-compatible format."""
    db = get_supabase_client()

    # Use provided user_id or default user for single-user development mode
    actual_user_id = user_id if user_id else DEFAULT_USER_ID

    result = db.table("stories").insert({
        "user_id": actual_user_id,
        "title": request.title,
        "premise": request.premise,
        "status": "active",
    }).execute()

    s = result.data[0]
    return FrontendStory(
        id=str(s["id"]),
        title=s["title"],
        createdAt=s["created_at"],
        lastPlayed=s["updated_at"],
    )


@router.get("/stories/{story_id}", response_model=FrontendStory)
async def get_story_details(story_id: UUID) -> FrontendStory:
    """Get detailed story information."""
    from schemas.api import WorldSetting, ContentBoundaries
    db = get_supabase_client()

    result = db.table("stories").select("*").eq(
        "id", str(story_id)
    ).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    s = result.data[0]

    # Parse world_setting from JSONB if present
    world_setting = None
    if s.get("world_setting"):
        ws = s["world_setting"]
        content_boundaries = None
        if ws.get("contentBoundaries"):
            cb = ws["contentBoundaries"]
            content_boundaries = ContentBoundaries(
                violenceLevel=cb.get("violenceLevel"),
                romanceLevel=cb.get("romanceLevel"),
                languageLevel=cb.get("languageLevel"),
            )
        world_setting = WorldSetting(
            worldType=ws.get("worldType"),
            settingDescription=ws.get("settingDescription"),
            themes=ws.get("themes"),
            protagonistType=ws.get("protagonistType"),
            protagonistDetails=ws.get("protagonistDetails"),
            narrativeStyle=ws.get("narrativeStyle"),
            pacing=ws.get("pacing"),
            contentBoundaries=content_boundaries,
            keyLocations=ws.get("keyLocations"),
            importantLore=ws.get("importantLore"),
            onboardingCompleted=ws.get("onboardingCompleted"),
        )

    return FrontendStory(
        id=str(s["id"]),
        title=s["title"],
        premise=s.get("premise"),
        description=s.get("description"),
        genre=s.get("genre"),
        tone=s.get("tone"),
        tags=s.get("tags") or [],
        grokInstructions=s.get("grok_instructions"),
        coverImageUrl=s.get("cover_image_url"),
        status=s.get("status"),
        isNsfw=s.get("is_nsfw"),
        worldSetting=world_setting,
        createdAt=s["created_at"],
        lastPlayed=s["updated_at"],
    )


@router.patch("/stories/{story_id}", response_model=FrontendStory)
async def update_story(story_id: UUID, data: UpdateStoryRequest) -> FrontendStory:
    """Update story settings including Grok instructions, NSFW, and world settings."""
    from schemas.api import WorldSetting, ContentBoundaries
    db = get_supabase_client()

    # Build update dict, mapping frontend camelCase to backend snake_case
    update_data = {}
    if data.title is not None:
        update_data["title"] = data.title
    if data.premise is not None:
        update_data["premise"] = data.premise
    if data.description is not None:
        update_data["description"] = data.description
    if data.genre is not None:
        update_data["genre"] = data.genre
    if data.tone is not None:
        update_data["tone"] = data.tone
    if data.tags is not None:
        update_data["tags"] = data.tags
    if data.grokInstructions is not None:
        update_data["grok_instructions"] = data.grokInstructions
    if data.coverImageUrl is not None:
        update_data["cover_image_url"] = data.coverImageUrl
    if data.status is not None:
        update_data["status"] = data.status
    if data.isNsfw is not None:
        update_data["is_nsfw"] = data.isNsfw
    if data.worldSetting is not None:
        # Convert Pydantic model to dict for JSONB storage
        update_data["world_setting"] = data.worldSetting.model_dump(exclude_none=True)

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Update the story
    update_data["updated_at"] = datetime.utcnow().isoformat()
    result = db.table("stories").update(update_data).eq(
        "id", str(story_id)
    ).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    s = result.data[0]

    # Parse world_setting from JSONB if present
    world_setting = None
    if s.get("world_setting"):
        ws = s["world_setting"]
        content_boundaries = None
        if ws.get("contentBoundaries"):
            cb = ws["contentBoundaries"]
            content_boundaries = ContentBoundaries(
                violenceLevel=cb.get("violenceLevel"),
                romanceLevel=cb.get("romanceLevel"),
                languageLevel=cb.get("languageLevel"),
            )
        world_setting = WorldSetting(
            worldType=ws.get("worldType"),
            settingDescription=ws.get("settingDescription"),
            themes=ws.get("themes"),
            protagonistType=ws.get("protagonistType"),
            protagonistDetails=ws.get("protagonistDetails"),
            narrativeStyle=ws.get("narrativeStyle"),
            pacing=ws.get("pacing"),
            contentBoundaries=content_boundaries,
            keyLocations=ws.get("keyLocations"),
            importantLore=ws.get("importantLore"),
            onboardingCompleted=ws.get("onboardingCompleted"),
        )

    return FrontendStory(
        id=str(s["id"]),
        title=s["title"],
        premise=s.get("premise"),
        description=s.get("description"),
        genre=s.get("genre"),
        tone=s.get("tone"),
        tags=s.get("tags") or [],
        grokInstructions=s.get("grok_instructions"),
        coverImageUrl=s.get("cover_image_url"),
        status=s.get("status"),
        isNsfw=s.get("is_nsfw"),
        worldSetting=world_setting,
        createdAt=s["created_at"],
        lastPlayed=s["updated_at"],
    )


# ============================================
# CHARACTERS
# ============================================


@router.get(
    "/stories/{story_id}/characters",
    response_model=FrontendCharacterListResponse
)
async def get_story_characters(
    story_id: UUID,
) -> FrontendCharacterListResponse:
    """Get all characters/entities for a story in frontend-compatible format."""
    db = get_supabase_client()

    # Verify story exists
    story_result = db.table("stories").select("id").eq(
        "id", str(story_id)
    ).execute()

    if not story_result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    # Get all entities
    entities_result = db.table("entities").select(
        "id, canonical_name, entity_type, description, computed_importance, metadata"
    ).eq("story_id", str(story_id)).is_(
        "deleted_at", "null"
    ).order("computed_importance", desc=True).execute()

    # Get character states for relationship meters
    char_states = db.table("character_states").select(
        "entity_id, stat_type, current_value, max_value, label"
    ).eq("story_id", str(story_id)).execute()

    # Build lookup of entity_id -> relationship meter
    relationship_meters = {}
    for state in char_states.data:
        entity_id = state["entity_id"]
        if entity_id not in relationship_meters:
            relationship_meters[entity_id] = RelationshipMeter(
                type=stat_type_to_relationship_type(state["stat_type"]),
                value=state["current_value"],
                label=state.get("label") or get_relationship_label(
                    state["current_value"], state["stat_type"]
                ),
            )

    # Get primary portraits
    portraits = db.table("entity_images").select(
        "entity_id, file_url, file_path"
    ).eq("is_primary", True).execute()

    portrait_lookup = {p["entity_id"]: p.get("file_url") or p.get("file_path") for p in portraits.data}

    # Transform entities to frontend format
    characters = []
    for e in entities_result.data:
        importance = importance_to_stars(e.get("computed_importance") or 0.5)

        # Determine type based on entity_type and importance
        entity_type = e["entity_type"]
        if entity_type == "location":
            frontend_type = "location"
        elif importance >= 4:
            frontend_type = "main"
        else:
            frontend_type = "side"

        character = FrontendCharacter(
            id=str(e["id"]),
            name=e["canonical_name"],
            type=frontend_type,
            portraitUrl=portrait_lookup.get(e["id"]),
            importance=importance,
            relationshipMeter=relationship_meters.get(e["id"]),
            description=e.get("description"),
        )
        characters.append(character)

    return FrontendCharacterListResponse(characters=characters)


@router.get(
    "/entities/{entity_id}",
    response_model=FrontendCharacterDetailsResponse
)
async def get_character_details(
    entity_id: UUID,
) -> FrontendCharacterDetailsResponse:
    """Get detailed character/entity info in frontend-compatible format."""
    db = get_supabase_client()

    # Get entity
    entity_result = db.table("entities").select(
        "id, story_id, canonical_name, entity_type, description, computed_importance"
    ).eq("id", str(entity_id)).execute()

    if not entity_result.data:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

    entity = entity_result.data[0]
    story_id = entity["story_id"]

    # Get facts
    facts_result = db.table("entity_facts").select(
        "fact_value"
    ).eq("entity_id", str(entity_id)).is_(
        "invalidated_session_id", "null"
    ).execute()

    facts = [f["fact_value"] for f in facts_result.data]

    # Get relationships
    rels_result = db.table("entity_relationships").select(
        "target_entity_id, relationship_type, description"
    ).eq("source_entity_id", str(entity_id)).is_(
        "ended_session_id", "null"
    ).execute()

    relationships = []
    for r in rels_result.data:
        # Get target entity name
        target = db.table("entities").select("canonical_name").eq(
            "id", r["target_entity_id"]
        ).execute()
        if target.data:
            relationships.append(CharacterRelationship(
                characterId=r["target_entity_id"],
                characterName=target.data[0]["canonical_name"],
                description=r.get("description") or r["relationship_type"],
            ))

    # Get event history (events where this entity participated)
    events_result = db.table("event_participants").select(
        "event_id"
    ).eq("entity_id", str(entity_id)).execute()

    event_history = []
    for ep in events_result.data[:10]:  # Limit to 10 events
        event = db.table("events").select(
            "description, created_at"
        ).eq("id", ep["event_id"]).execute()
        if event.data:
            event_history.append(CharacterEvent(
                timestamp=event.data[0]["created_at"],
                event=event.data[0]["description"][:100],
            ))

    # Get relationship meter
    char_state = db.table("character_states").select(
        "stat_type, current_value, max_value, label"
    ).eq("entity_id", str(entity_id)).limit(1).execute()

    relationship_meter = None
    if char_state.data:
        state = char_state.data[0]
        relationship_meter = RelationshipMeter(
            type=stat_type_to_relationship_type(state["stat_type"]),
            value=state["current_value"],
            label=state.get("label") or get_relationship_label(
                state["current_value"], state["stat_type"]
            ),
        )

    # Get portrait
    portrait = db.table("entity_images").select(
        "file_url, file_path"
    ).eq("entity_id", str(entity_id)).eq("is_primary", True).execute()

    portrait_url = None
    if portrait.data:
        portrait_url = portrait.data[0].get("file_url") or portrait.data[0].get("file_path")

    importance = importance_to_stars(entity.get("computed_importance") or 0.5)
    entity_type = entity["entity_type"]

    if entity_type == "location":
        frontend_type = "location"
    elif importance >= 4:
        frontend_type = "main"
    else:
        frontend_type = "side"

    character = FrontendCharacter(
        id=str(entity["id"]),
        name=entity["canonical_name"],
        type=frontend_type,
        portraitUrl=portrait_url,
        importance=importance,
        relationshipMeter=relationship_meter,
        description=entity.get("description"),
        facts=facts if facts else None,
        relationships=relationships if relationships else None,
        eventHistory=event_history if event_history else None,
    )

    return FrontendCharacterDetailsResponse(character=character)


# ============================================
# GAME STATE
# ============================================


@router.get(
    "/stories/{story_id}/state",
    response_model=FrontendGameStateResponse
)
async def get_game_state_frontend(
    story_id: UUID,
) -> FrontendGameStateResponse:
    """Get game state in frontend-compatible format."""
    db = get_supabase_client()

    # Verify story exists
    story_result = db.table("stories").select("id, current_situation").eq(
        "id", str(story_id)
    ).execute()

    if not story_result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    story = story_result.data[0]

    # Get protagonist stats
    stats_result = db.table("protagonist_stats").select(
        "stat_name, current_value, max_value"
    ).eq("story_id", str(story_id)).execute()

    stats = []
    for s in stats_result.data:
        try:
            current = int(float(s["current_value"])) if s["current_value"] else 0
            max_val = int(float(s["max_value"])) if s["max_value"] else 100
        except (ValueError, TypeError):
            current = 0
            max_val = 100

        stats.append(FrontendStat(
            name=s["stat_name"],
            current=current,
            max=max_val,
        ))

    # Get skills
    skills_result = db.table("protagonist_skills").select(
        "skill_name, rank"
    ).eq("story_id", str(story_id)).execute()

    skills = [
        FrontendSkill(
            name=s["skill_name"],
            rank=s.get("rank") or "E",
        )
        for s in skills_result.data
    ]

    # Get inventory
    inventory_result = db.table("protagonist_inventory").select(
        "id, item_name, description"
    ).eq("story_id", str(story_id)).is_(
        "lost_session_id", "null"
    ).execute()

    inventory = [
        FrontendInventoryItem(
            id=str(i["id"]),
            name=i["item_name"],
            description=i.get("description") or "",
        )
        for i in inventory_result.data
    ]

    # Get status effects
    effects_result = db.table("protagonist_status_effects").select(
        "effect_name"
    ).eq("story_id", str(story_id)).is_(
        "removed_session_id", "null"
    ).execute()

    status_effects = [e["effect_name"] for e in effects_result.data]

    # Get NPC relationships
    char_states_result = db.table("character_states").select(
        "entity_id, stat_type, current_value, max_value, label"
    ).eq("story_id", str(story_id)).execute()

    npc_relationships = []
    for state in char_states_result.data:
        # Get entity details
        entity = db.table("entities").select(
            "canonical_name"
        ).eq("id", state["entity_id"]).execute()

        if entity.data:
            # Get portrait
            portrait = db.table("entity_images").select(
                "file_url, file_path"
            ).eq("entity_id", state["entity_id"]).eq("is_primary", True).execute()

            portrait_url = None
            if portrait.data:
                portrait_url = portrait.data[0].get("file_url") or portrait.data[0].get("file_path")

            npc_relationships.append(FrontendNPCRelationship(
                id=state["entity_id"],
                name=entity.data[0]["canonical_name"],
                portraitUrl=portrait_url,
                relationshipMeter=RelationshipMeter(
                    type=stat_type_to_relationship_type(state["stat_type"]),
                    value=state["current_value"],
                    label=state.get("label") or get_relationship_label(
                        state["current_value"], state["stat_type"]
                    ),
                ),
            ))

    # Build world state (basic implementation - can be enhanced)
    world_state = FrontendWorldState(
        location=FrontendLocation(
            name=story.get("current_situation") or "Unknown Location",
        ),
        time="Day 1",  # TODO: Track time in database
        quests=[],  # TODO: Add quests table
    )

    protagonist = FrontendProtagonist(
        stats=stats,
        skills=skills,
        inventory=inventory,
        statusEffects=status_effects,
    )

    game_state = FrontendGameState(
        protagonist=protagonist,
        npcRelationships=npc_relationships,
        worldState=world_state,
    )

    return FrontendGameStateResponse(gameState=game_state)


# ============================================
# PORTRAIT GENERATION
# ============================================


@router.post(
    "/entities/{entity_id}/generate-portrait",
    response_model=GeneratePortraitResponse
)
async def generate_portrait_frontend(
    entity_id: UUID,
    request: GeneratePortraitRequest,
) -> GeneratePortraitResponse:
    """Generate a portrait for an entity using ComfyUI."""
    from services.image_generation import ImageGenerationService, CharacterConsistencyService
    from schemas.extraction import Entity, EntityType, Fact, FactType

    db = get_supabase_client()

    # Get entity with full details
    entity_result = db.table("entities").select(
        "id, story_id, canonical_name, entity_type, description, computed_importance"
    ).eq("id", str(entity_id)).execute()

    if not entity_result.data:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

    entity_data = entity_result.data[0]
    story_id = UUID(entity_data["story_id"])

    # Get facts for this entity
    facts_result = db.table("entity_facts").select(
        "fact_type, fact_value, importance"
    ).eq("entity_id", str(entity_id)).is_(
        "invalidated_session_id", "null"
    ).execute()

    facts = [
        Fact(
            fact_type=FactType(f["fact_type"]) if f["fact_type"] in [e.value for e in FactType] else FactType.TRAIT,
            fact_value=f["fact_value"],
            importance=f.get("importance", 0.5)
        )
        for f in facts_result.data
    ]

    # Get aliases
    aliases_result = db.table("entity_aliases").select("alias").eq(
        "entity_id", str(entity_id)
    ).execute()
    aliases = [a["alias"] for a in aliases_result.data]

    # Build Entity model
    entity = Entity(
        entity_type=EntityType(entity_data["entity_type"]) if entity_data["entity_type"] in [e.value for e in EntityType] else EntityType.CHARACTER,
        canonical_name=entity_data["canonical_name"],
        aliases=aliases,
        description=entity_data.get("description", ""),
        facts=facts,
        importance=entity_data.get("computed_importance", 0.5),
    )

    # Generate portrait using ComfyUI
    try:
        image_service = ImageGenerationService()
        consistency_service = CharacterConsistencyService(image_service=image_service, db_client=db)

        # Convert custom_loras from Pydantic models to dicts if present
        custom_loras_dict = None
        if request.custom_loras:
            custom_loras_dict = [{"name": l.name, "weight": l.weight} for l in request.custom_loras]

        # Debug logging
        print(f"[Portrait] Entity: {entity.canonical_name}")
        print(f"[Portrait] Request params: width={request.width}, height={request.height}, steps={request.steps}, cfg={request.cfg}")
        print(f"[Portrait] HiRes: scale={request.hires_scale}, denoise={request.hires_denoise}")
        print(f"[Portrait] LoRAs: custom={custom_loras_dict}, use_standard={request.use_standard_lora}, use_default={request.use_default_loras}")
        print(f"[Portrait] Additional tags: {request.additionalTags}")
        print(f"[Portrait] Pose: {request.pose}")

        result = await consistency_service.generate_consistent_portrait(
            entity=entity,
            entity_id=entity_id,
            story_id=story_id,
            style=request.style or "anime",
            width=request.width,
            height=request.height,
            seed=request.seed,
            pose=request.pose,
            expression=request.expression,
            additional_tags=request.additionalTags,
            # Extended HiRes and LoRA parameters
            custom_loras=custom_loras_dict,
            use_standard_lora=request.use_standard_lora,
            use_default_loras=request.use_default_loras,
            steps=request.steps,
            cfg=request.cfg,
            hires_scale=request.hires_scale,
            hires_denoise=request.hires_denoise,
        )

        # Return the generated image URL/path
        portrait_url = result.file_url or result.file_path

        # Debug: Log the generated prompt
        print(f"[Portrait] Generated prompt: {result.generation_prompt[:500]}...")
        print(f"[Portrait] Final dimensions: {result.width}x{result.height}")
        print(f"[Portrait] Seed: {result.seed}")

        # Convert loras_applied to the expected format
        loras_applied = None
        if hasattr(result, 'loras_applied') and result.loras_applied:
            loras_applied = result.loras_applied

        return GeneratePortraitResponse(
            portraitUrl=portrait_url,
            prompt=result.generation_prompt,
            seed=result.seed,
            model=result.model_used,
            lorasApplied=loras_applied,
        )

    except Exception as e:
        print(f"[ImageGen] Portrait generation error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to DiceBear on error
        seed_str = str(request.seed) if request.seed else entity_data["canonical_name"].lower().replace(" ", "-")
        portrait_url = f"https://api.dicebear.com/7.x/avataaars/svg?seed={seed_str}"
        return GeneratePortraitResponse(portraitUrl=portrait_url)


# ============================================
# REAL-TIME CHAT
# ============================================


class SendMessageRequest(BaseModel):
    """Request to send a chat message."""
    message: str


class ChatMessageResponse(BaseModel):
    """A single chat message in frontend format."""
    id: str
    type: str  # narrator, character, player-choice, system, image
    content: str
    characterId: Optional[str] = None
    characterName: Optional[str] = None
    characterPortraitUrl: Optional[str] = None
    imageUrl: Optional[str] = None
    timestamp: str


class IntroducedNPCResponse(BaseModel):
    """An NPC that was introduced in the narration."""
    name: str
    apparentAge: Optional[int] = None
    physicalDescription: str
    clothingDescription: Optional[str] = None
    personalityHints: Optional[str] = None
    role: Optional[str] = None
    importance: float = 0.5
    # If auto-saved to database
    entityId: Optional[str] = None


class ProtagonistStatusResponse(BaseModel):
    """Current status of the protagonist."""
    health: Optional[int] = None
    stamina: Optional[int] = None
    arousal: Optional[int] = None
    stress: Optional[int] = None
    hunger: Optional[int] = None
    customStats: Optional[dict[str, int]] = None
    statusEffects: Optional[list[str]] = None


class WorldStateUpdateResponse(BaseModel):
    """World state changes from the narration."""
    currentLocation: Optional[str] = None
    timeOfDay: Optional[str] = None
    discoveredFacts: Optional[list[str]] = None
    unlockedLocations: Optional[list[str]] = None


class SendMessageResponse(BaseModel):
    """Response from sending a chat message."""
    messages: list[ChatMessageResponse]
    # Scene metadata for image generation
    sceneImageSuggestion: Optional[str] = None
    currentMood: Optional[str] = None
    charactersPresent: Optional[list[str]] = None
    # Auto-triggered scene generation task (if scene_image_suggestion was present)
    sceneGenerationTaskId: Optional[str] = None
    # New immersive storytelling fields
    npcsIntroduced: Optional[list[IntroducedNPCResponse]] = None
    protagonistStatus: Optional[ProtagonistStatusResponse] = None
    worldStateUpdate: Optional[WorldStateUpdateResponse] = None
    asksProtagonistInfo: Optional[str] = None


@router.post(
    "/sessions/{story_id}/send",
    response_model=SendMessageResponse
)
async def send_chat_message(
    story_id: UUID,
    request: SendMessageRequest,
) -> SendMessageResponse:
    """
    Send a message to the story chat and get AI response.

    Uses Grok to generate narrative responses based on story context,
    including semantic memories, relationship meters, and physical states.
    Returns the player's message and the narrator's response(s).
    """
    from services.chat import get_chat_service
    from api.dependencies import get_context_service
    import uuid as uuid_lib

    db = get_supabase_client()

    # Get story info including grok_instructions, is_nsfw, genre, tags
    story_result = db.table("stories").select(
        "id, title, premise, current_situation, grok_instructions, is_nsfw, genre, tags"
    ).eq("id", str(story_id)).execute()

    if not story_result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    story = story_result.data[0]

    # Get characters for context with relationship info
    chars_result = db.table("entities").select(
        "id, canonical_name, description, entity_type"
    ).eq("story_id", str(story_id)).eq(
        "entity_type", "character"
    ).limit(5).execute()

    characters = []
    char_names = []
    for c in chars_result.data:
        char_names.append(c["canonical_name"])
        # Get relationship summary if available
        rel_result = db.table("intimacy_metrics").select(
            "affection, trust, lust, comfort"
        ).eq("story_id", str(story_id)).eq(
            "character", c["canonical_name"]
        ).order("created_at", desc=True).limit(1).execute()

        rel_summary = "New acquaintance"
        if rel_result.data:
            r = rel_result.data[0]
            if r.get("affection", 0) > 70:
                rel_summary = "Close relationship"
            elif r.get("affection", 0) > 40:
                rel_summary = "Friendly"
            elif r.get("trust", 0) < 30:
                rel_summary = "Wary"

        characters.append({
            "name": c["canonical_name"],
            "description": c.get("description", ""),
            "relationship_summary": rel_summary,
        })

    # Get recent events for context
    events_result = db.table("events").select(
        "description"
    ).eq("story_id", str(story_id)).order(
        "created_at", desc=True
    ).limit(5).execute()

    recent_events = [e["description"][:200] for e in events_result.data]

    # Get relationship meters for present characters
    relationship_meters = []
    for char_name in char_names:
        rel_result = db.table("intimacy_metrics").select(
            "character, affection, trust, lust, comfort, jealousy"
        ).eq("story_id", str(story_id)).eq(
            "character", char_name
        ).order("created_at", desc=True).limit(1).execute()

        if rel_result.data:
            r = rel_result.data[0]
            relationship_meters.append({
                "name": r["character"],
                "affection": r.get("affection"),
                "trust": r.get("trust"),
                "lust": r.get("lust"),
                "comfort": r.get("comfort"),
            })

    # Get physical states for present characters
    physical_states = []
    for char_name in char_names:
        state_result = db.table("physical_states").select(
            "character, clothing, position, location_in_scene, temporary_states"
        ).eq("story_id", str(story_id)).eq(
            "character", char_name
        ).order("created_at", desc=True).limit(1).execute()

        if state_result.data:
            s = state_result.data[0]
            physical_states.append({
                "character": s["character"],
                "clothing": s.get("clothing") or [],
                "position": s.get("position"),
                "location_in_scene": s.get("location_in_scene"),
                "temporary_states": s.get("temporary_states") or [],
            })

    # Get current scene state
    scene_state = None
    scene_result = db.table("scene_states").select(
        "scene_type, scene_active, participants, mood, interrupted"
    ).eq("story_id", str(story_id)).order(
        "created_at", desc=True
    ).limit(1).execute()

    if scene_result.data:
        s = scene_result.data[0]
        scene_state = {
            "type": s.get("scene_type", "dialogue"),
            "mood": s.get("mood", "neutral"),
            "participants": s.get("participants") or char_names,
        }

    # Get relevant semantic memories using SpicyChat pattern
    context_service = get_context_service()
    semantic_memories = await context_service.get_semantic_memories(
        story_id=story_id,
        characters_present=char_names,
        current_emotion=scene_state.get("mood") if scene_state else None,
        max_memories=15,
    )

    # Format current situation as dict for context builder
    current_situation = None
    if story.get("current_situation"):
        current_situation = {
            "location": "Current Scene",
            "time": "",
            "description": story.get("current_situation"),
        }

    # Generate response using Grok with full context
    # Use is_nsfw from story settings (defaults to True if not set)
    is_nsfw = story.get("is_nsfw") if story.get("is_nsfw") is not None else True
    chat_service = get_chat_service()
    response = chat_service.generate_response(
        player_message=request.message,
        story_premise=story.get("premise") or story["title"],
        story_title=story.get("title"),
        story_genre=story.get("genre"),
        story_tags=story.get("tags"),
        current_situation=current_situation,
        characters_present=characters,
        recent_events=recent_events,
        relationship_meters=relationship_meters,
        physical_states=physical_states,
        semantic_memories=semantic_memories,
        scene_state=scene_state,
        is_nsfw=is_nsfw,
        custom_instructions=story.get("grok_instructions"),
    )

    # Build response messages
    now = datetime.now().isoformat()
    messages = []

    # Player's message
    messages.append(ChatMessageResponse(
        id=str(uuid_lib.uuid4()),
        type="player-choice",
        content=request.message,
        timestamp=now,
    ))

    # Narrator response
    messages.append(ChatMessageResponse(
        id=str(uuid_lib.uuid4()),
        type="narrator",
        content=response.narration,
        timestamp=now,
    ))

    # Character dialogue if present
    if response.character_dialogue and response.speaking_character:
        # Try to find character portrait
        char_portrait = None
        for c in chars_result.data:
            if c["canonical_name"].lower() == response.speaking_character.lower():
                # Get portrait
                portrait = db.table("entity_images").select(
                    "file_url, file_path"
                ).eq("entity_id", c["id"]).eq("is_primary", True).execute()
                if portrait.data:
                    portrait_path = portrait.data[0].get("file_url") or portrait.data[0].get("file_path")
                    # Ensure proper URL format - prefix with / if it's a relative path
                    if portrait_path and not portrait_path.startswith(("http://", "https://", "/")):
                        char_portrait = f"/{portrait_path}"
                    else:
                        char_portrait = portrait_path
                break

        messages.append(ChatMessageResponse(
            id=str(uuid_lib.uuid4()),
            type="character",
            content=response.character_dialogue,
            characterName=response.speaking_character,
            characterPortraitUrl=char_portrait,
            timestamp=now,
        ))

    # Suggested choices as system message
    if response.suggested_choices:
        choices_text = "Choose your action:\n" + "\n".join([
            f"• {choice}" for choice in response.suggested_choices
        ])
        messages.append(ChatMessageResponse(
            id=str(uuid_lib.uuid4()),
            type="system",
            content=choices_text,
            timestamp=now,
        ))

    # Persist messages to database
    for msg in messages:
        try:
            db.table("chat_messages").insert({
                "id": msg.id,
                "story_id": str(story_id),
                "message_type": msg.type,
                "content": msg.content,
                "character_name": msg.characterName,
                "character_portrait_url": msg.characterPortraitUrl,
                "image_url": msg.imageUrl,
            }).execute()
        except Exception as e:
            print(f"[Chat] Failed to persist message: {e}")

    # Auto-trigger scene generation if AI suggested a scene image
    scene_task_id = None
    if response.scene_image_suggestion:
        try:
            from services.scene_generation import get_scene_generation_service
            scene_service = get_scene_generation_service(db_client=db)

            # Trigger async scene generation with physical states for LoRA mapping
            scene_result = await scene_service.generate_scene_async(
                story_id=story_id,
                scene_description=response.scene_image_suggestion,
                characters_present=char_names,
                physical_states=physical_states,
                mood=response.mood,
                wait_for_result=False,  # Non-blocking
            )
            scene_task_id = scene_result.get("task_id")
            print(f"[Chat] Auto-triggered scene generation: task_id={scene_task_id}")
        except Exception as e:
            print(f"[Chat] Failed to trigger scene generation: {e}")
            # Non-fatal - don't fail the chat response

    return SendMessageResponse(
        messages=messages,
        sceneImageSuggestion=response.scene_image_suggestion,
        currentMood=response.mood,
        charactersPresent=char_names,
        sceneGenerationTaskId=scene_task_id,
    )


@router.post(
    "/sessions/{story_id}/start",
    response_model=SendMessageResponse
)
async def start_story_session(
    story_id: UUID,
) -> SendMessageResponse:
    """
    Start a new story session and generate IMMERSIVE opening narration.

    Uses the enhanced immersive opening system that:
    - Grounds the player through sensory experience, not exposition
    - Introduces NPCs with full physical descriptions
    - Discovers the world organically through action
    - Asks for protagonist details naturally through characters
    - Auto-saves introduced NPCs to the database
    """
    from services.chat import get_chat_service
    import uuid as uuid_lib

    db = get_supabase_client()

    # Get story info INCLUDING world_setting for rich context
    story_result = db.table("stories").select(
        "id, title, premise, current_situation, grok_instructions, is_nsfw, genre, tags, world_setting"
    ).eq("id", str(story_id)).execute()

    if not story_result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    story = story_result.data[0]

    # Get main characters (may be empty for new stories)
    chars_result = db.table("entities").select(
        "id, canonical_name, description"
    ).eq("story_id", str(story_id)).eq(
        "entity_type", "character"
    ).order("computed_importance", desc=True).limit(5).execute()

    characters = [
        {"name": c["canonical_name"], "description": c.get("description", "")}
        for c in chars_result.data
    ]

    # Use is_nsfw from story settings (defaults to True if not set)
    is_nsfw = story.get("is_nsfw") if story.get("is_nsfw") is not None else True

    # Extract protagonist details from world_setting if available
    world_setting = story.get("world_setting") or {}
    protagonist_details = world_setting.get("protagonistDetails")

    # Generate IMMERSIVE opening with full world context
    chat_service = get_chat_service()
    response = chat_service.generate_opening(
        story_premise=story.get("premise") or story["title"],
        story_title=story.get("title"),
        story_genre=story.get("genre"),
        story_tags=story.get("tags"),
        characters=characters,
        starting_location=story.get("current_situation"),
        is_nsfw=is_nsfw,
        custom_instructions=story.get("grok_instructions"),
        world_setting=world_setting,
        protagonist_details=protagonist_details,
    )

    # Build response messages
    now = datetime.now().isoformat()
    messages = []

    # System welcome
    messages.append(ChatMessageResponse(
        id=str(uuid_lib.uuid4()),
        type="system",
        content=f"Welcome to {story['title']}",
        timestamp=now,
    ))

    # Opening narration
    messages.append(ChatMessageResponse(
        id=str(uuid_lib.uuid4()),
        type="narrator",
        content=response.narration,
        timestamp=now,
    ))

    # If AI is asking for protagonist info (like name), add that as a system message
    if response.asks_protagonist_info:
        messages.append(ChatMessageResponse(
            id=str(uuid_lib.uuid4()),
            type="system",
            content=f"💭 {response.asks_protagonist_info}",
            timestamp=now,
        ))

    # Suggested choices
    if response.suggested_choices:
        choices_text = "What will you do?\n" + "\n".join([
            f"• {choice}" for choice in response.suggested_choices
        ])
        messages.append(ChatMessageResponse(
            id=str(uuid_lib.uuid4()),
            type="system",
            content=choices_text,
            timestamp=now,
        ))

    # AUTO-SAVE INTRODUCED NPCs to database
    saved_npcs = []
    if response.npcs_introduced:
        for npc in response.npcs_introduced:
            try:
                # Build description from physical details
                description_parts = [npc.physical_description]
                if npc.clothing_description:
                    description_parts.append(f"Wearing: {npc.clothing_description}")
                if npc.personality_hints:
                    description_parts.append(f"Personality: {npc.personality_hints}")
                full_description = " ".join(description_parts)

                # Check if entity already exists
                existing = db.table("entities").select("id").eq(
                    "story_id", str(story_id)
                ).eq("canonical_name", npc.name).execute()

                if existing.data:
                    # Entity already exists - add to saved_npcs with existing ID
                    saved_npcs.append({
                        "id": existing.data[0]["id"],
                        "name": npc.name,
                        "description": full_description,
                    })
                    print(f"[Opening] NPC already exists: {npc.name} (id: {existing.data[0]['id']})")
                else:
                    # Create new entity
                    entity_result = db.table("entities").insert({
                        "story_id": str(story_id),
                        "canonical_name": npc.name,
                        "entity_type": "character",
                        "description": full_description,
                        "computed_importance": npc.importance,
                        "metadata": {
                            "apparent_age": npc.apparent_age,
                            "role": npc.role,
                            "physical_description": npc.physical_description,
                            "clothing": npc.clothing_description,
                            "personality_hints": npc.personality_hints,
                            "introduced_at": now,
                        }
                    }).execute()

                    if entity_result.data:
                        saved_npcs.append({
                            "id": entity_result.data[0]["id"],
                            "name": npc.name,
                            "description": full_description,
                        })
                        print(f"[Opening] Auto-saved NPC: {npc.name}")

            except Exception as e:
                print(f"[Opening] Failed to save NPC {npc.name}: {e}")

    # Update char_names to include newly introduced NPCs
    char_names = [c["name"] for c in characters]
    for npc in (response.npcs_introduced or []):
        if npc.name not in char_names:
            char_names.append(npc.name)

    # Persist messages to database
    for msg in messages:
        try:
            db.table("chat_messages").insert({
                "id": msg.id,
                "story_id": str(story_id),
                "message_type": msg.type,
                "content": msg.content,
                "character_name": msg.characterName,
                "character_portrait_url": msg.characterPortraitUrl,
                "image_url": msg.imageUrl,
            }).execute()
        except Exception as e:
            print(f"[Chat] Failed to persist message: {e}")

    # Auto-trigger scene generation if AI suggested a scene image for opening
    scene_task_id = None
    if response.scene_image_suggestion:
        try:
            from services.scene_generation import get_scene_generation_service
            scene_service = get_scene_generation_service(db_client=db)

            # For opening, we don't have physical states yet, just use defaults
            scene_result = await scene_service.generate_scene_async(
                story_id=story_id,
                scene_description=response.scene_image_suggestion,
                characters_present=char_names,
                physical_states=None,
                mood=response.mood,
                wait_for_result=False,
            )
            scene_task_id = scene_result.get("task_id")
            print(f"[Chat] Auto-triggered opening scene generation: task_id={scene_task_id}")
        except Exception as e:
            print(f"[Chat] Failed to trigger opening scene generation: {e}")

    # Build NPC response objects
    npcs_response = None
    if response.npcs_introduced:
        npcs_response = []
        for npc in response.npcs_introduced:
            # Find entity ID if it was saved
            entity_id = None
            for saved in saved_npcs:
                if saved["name"] == npc.name:
                    entity_id = saved["id"]
                    break

            npcs_response.append(IntroducedNPCResponse(
                name=npc.name,
                apparentAge=npc.apparent_age,
                physicalDescription=npc.physical_description,
                clothingDescription=npc.clothing_description,
                personalityHints=npc.personality_hints,
                role=npc.role,
                importance=npc.importance,
                entityId=entity_id,
            ))

    # Build protagonist status response
    protagonist_status_response = None
    if response.protagonist_status:
        ps = response.protagonist_status
        protagonist_status_response = ProtagonistStatusResponse(
            health=ps.health,
            stamina=ps.stamina,
            arousal=ps.arousal,
            stress=ps.stress,
            hunger=ps.hunger,
            customStats=ps.custom_stats,
            statusEffects=ps.status_effects,
        )

    # Build world state response
    world_state_response = None
    if response.world_state_update:
        ws = response.world_state_update
        world_state_response = WorldStateUpdateResponse(
            currentLocation=ws.current_location,
            timeOfDay=ws.time_of_day,
            discoveredFacts=ws.discovered_facts,
            unlockedLocations=ws.unlocked_locations,
        )

    return SendMessageResponse(
        messages=messages,
        sceneImageSuggestion=response.scene_image_suggestion,
        currentMood=response.mood,
        charactersPresent=char_names,
        sceneGenerationTaskId=scene_task_id,
        npcsIntroduced=npcs_response,
        protagonistStatus=protagonist_status_response,
        worldStateUpdate=world_state_response,
        asksProtagonistInfo=response.asks_protagonist_info,
    )


@router.get(
    "/sessions/{story_id}/context",
    response_model=SendMessageResponse
)
async def get_session_context(story_id: UUID) -> SendMessageResponse:
    """
    Get existing chat messages for a story session.

    Returns all persisted chat messages in chronological order.
    Called on page load to restore chat history.
    """
    db = get_supabase_client()

    # Verify story exists
    story_result = db.table("stories").select("id").eq(
        "id", str(story_id)
    ).execute()

    if not story_result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    # Get all messages for this story, ordered by creation time
    result = db.table("chat_messages").select("*").eq(
        "story_id", str(story_id)
    ).order("created_at").execute()

    messages = [
        ChatMessageResponse(
            id=str(m["id"]),
            type=m["message_type"],
            content=m["content"],
            characterName=m.get("character_name"),
            characterPortraitUrl=m.get("character_portrait_url"),
            imageUrl=m.get("image_url"),
            timestamp=m["created_at"],
        )
        for m in result.data
    ]

    return SendMessageResponse(messages=messages)


# ============================================
# MEMORY MANAGEMENT
# ============================================


@router.get(
    "/stories/{story_id}/memories",
    response_model=SemanticMemoryListResponse
)
async def get_story_memories(
    story_id: UUID,
    include_hidden: bool = False,
    pinned_only: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> SemanticMemoryListResponse:
    """
    Get semantic memories for a story.

    User-editable memory system that allows pinning, editing, and hiding.
    """
    from schemas.api import FrontendSemanticMemory, SemanticMemoryListResponse

    db = get_supabase_client()

    # Build query
    query = db.table("semantic_memories").select("*").eq(
        "story_id", str(story_id)
    )

    if not include_hidden:
        query = query.eq("hidden", False)

    if pinned_only:
        query = query.eq("pinned", True)

    # Get total count
    count_result = db.table("semantic_memories").select(
        "id", count="exact"
    ).eq("story_id", str(story_id)).execute()
    total = count_result.count or 0

    # Get memories with pagination
    result = query.order("pinned", desc=True).order(
        "importance", desc=True
    ).order("created_at", desc=True).range(offset, offset + limit - 1).execute()

    memories = [
        FrontendSemanticMemory(
            id=str(m["id"]),
            memoryText=m["memory_text"],
            charactersInvolved=m.get("characters_involved") or [],
            primaryEmotion=m["primary_emotion"],
            topics=m.get("topics") or [],
            importance=m.get("importance", 0.5),
            pinned=m.get("pinned", False),
            userEdited=m.get("user_edited", False),
            hidden=m.get("hidden", False),
            setupForPayoff=m.get("setup_for_payoff", False),
            createdAt=m["created_at"],
            sessionId=str(m["session_id"]) if m.get("session_id") else None,
        )
        for m in result.data
    ]

    return SemanticMemoryListResponse(memories=memories, total=total)


@router.patch(
    "/stories/{story_id}/memories/{memory_id}",
    response_model=FrontendSemanticMemory
)
async def update_memory(
    story_id: UUID,
    memory_id: UUID,
    request: UpdateMemoryRequest,
) -> FrontendSemanticMemory:
    """
    Update a semantic memory (pin, edit, hide, adjust importance).
    """
    from schemas.api import FrontendSemanticMemory, UpdateMemoryRequest

    db = get_supabase_client()

    # Build update dict
    update_data = {"updated_at": datetime.utcnow().isoformat()}

    if request.memoryText is not None:
        update_data["memory_text"] = request.memoryText
        update_data["user_edited"] = True

    if request.pinned is not None:
        update_data["pinned"] = request.pinned

    if request.hidden is not None:
        update_data["hidden"] = request.hidden

    if request.importance is not None:
        update_data["importance"] = request.importance

    # Update the memory
    result = db.table("semantic_memories").update(update_data).eq(
        "id", str(memory_id)
    ).eq("story_id", str(story_id)).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Memory not found")

    m = result.data[0]
    return FrontendSemanticMemory(
        id=str(m["id"]),
        memoryText=m["memory_text"],
        charactersInvolved=m.get("characters_involved") or [],
        primaryEmotion=m["primary_emotion"],
        topics=m.get("topics") or [],
        importance=m.get("importance", 0.5),
        pinned=m.get("pinned", False),
        userEdited=m.get("user_edited", False),
        hidden=m.get("hidden", False),
        setupForPayoff=m.get("setup_for_payoff", False),
        createdAt=m["created_at"],
        sessionId=str(m["session_id"]) if m.get("session_id") else None,
    )


@router.post(
    "/stories/{story_id}/memories",
    response_model=FrontendSemanticMemory
)
async def create_memory(
    story_id: UUID,
    request: CreateMemoryRequest,
) -> FrontendSemanticMemory:
    """
    Manually create a semantic memory.

    Useful for adding important context the AI might have missed.
    """
    from schemas.api import FrontendSemanticMemory, CreateMemoryRequest
    import json

    db = get_supabase_client()

    # Verify story exists
    story_result = db.table("stories").select("id").eq(
        "id", str(story_id)
    ).execute()

    if not story_result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    # Get latest session for the story (or create dummy reference)
    session_result = db.table("sessions").select("id").eq(
        "story_id", str(story_id)
    ).order("session_number", desc=True).limit(1).execute()

    session_id = session_result.data[0]["id"] if session_result.data else None

    # If no session exists, we need to create a placeholder
    if not session_id:
        # Create a placeholder session
        new_session = db.table("sessions").insert({
            "story_id": str(story_id),
            "session_number": 0,
            "status": "processed",
            "summary": "Manual memory entry session",
        }).execute()
        session_id = new_session.data[0]["id"]

    # Create the memory
    now = datetime.utcnow().isoformat()
    result = db.table("semantic_memories").insert({
        "story_id": str(story_id),
        "session_id": session_id,
        "memory_text": request.memoryText,
        "characters_involved": json.dumps(request.charactersInvolved),
        "primary_emotion": request.primaryEmotion,
        "topics": json.dumps(request.topics),
        "importance": request.importance,
        "pinned": request.pinned,
        "user_edited": True,
        "created_at": now,
        "updated_at": now,
    }).execute()

    m = result.data[0]
    return FrontendSemanticMemory(
        id=str(m["id"]),
        memoryText=m["memory_text"],
        charactersInvolved=request.charactersInvolved,
        primaryEmotion=m["primary_emotion"],
        topics=request.topics,
        importance=m.get("importance", 0.5),
        pinned=m.get("pinned", False),
        userEdited=True,
        hidden=False,
        setupForPayoff=False,
        createdAt=m["created_at"],
        sessionId=str(session_id),
    )


@router.delete("/stories/{story_id}/memories/{memory_id}")
async def delete_memory(story_id: UUID, memory_id: UUID):
    """
    Delete a semantic memory (hard delete).
    """
    db = get_supabase_client()

    result = db.table("semantic_memories").delete().eq(
        "id", str(memory_id)
    ).eq("story_id", str(story_id)).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {"success": True, "deleted_id": str(memory_id)}


# ============================================
# INTIMACY METRICS
# ============================================


@router.get(
    "/stories/{story_id}/intimacy",
    response_model=IntimacyMetricsListResponse
)
async def get_intimacy_metrics(story_id: UUID) -> IntimacyMetricsListResponse:
    """
    Get multi-dimensional intimacy metrics for all characters in a story.
    """
    from schemas.api import FrontendIntimacyMetrics, IntimacyMetricsListResponse

    db = get_supabase_client()

    # Get intimacy metrics with entity info
    result = db.table("intimacy_metrics").select(
        "*, entities!inner(id, canonical_name)"
    ).eq("story_id", str(story_id)).execute()

    # Get portraits
    entity_ids = [m["entity_id"] for m in result.data]
    portraits = {}
    if entity_ids:
        portrait_result = db.table("entity_images").select(
            "entity_id, file_url, file_path"
        ).in_("entity_id", entity_ids).eq("is_primary", True).execute()
        portraits = {
            p["entity_id"]: p.get("file_url") or p.get("file_path")
            for p in portrait_result.data
        }

    metrics = [
        FrontendIntimacyMetrics(
            characterId=str(m["entity_id"]),
            characterName=m["entities"]["canonical_name"],
            affection=m.get("affection", 50),
            trust=m.get("trust", 50),
            lust=m.get("lust", 0),
            comfort=m.get("comfort", 50),
            jealousy=m.get("jealousy", 0),
            submission=m.get("submission"),
            dominance=m.get("dominance"),
            portraitUrl=portraits.get(m["entity_id"]),
        )
        for m in result.data
    ]

    return IntimacyMetricsListResponse(metrics=metrics)


@router.get(
    "/stories/{story_id}/intimacy/{entity_id}/history",
    response_model=IntimacyHistoryResponse
)
async def get_intimacy_history(
    story_id: UUID,
    entity_id: UUID,
    limit: int = 20,
) -> IntimacyHistoryResponse:
    """
    Get the history of intimacy changes for a specific character.
    """
    from schemas.api import IntimacyHistoryEntry, IntimacyHistoryResponse

    db = get_supabase_client()

    # Get entity name
    entity_result = db.table("entities").select("canonical_name").eq(
        "id", str(entity_id)
    ).execute()

    if not entity_result.data:
        raise HTTPException(status_code=404, detail="Character not found")

    character_name = entity_result.data[0]["canonical_name"]

    # Get intimacy metric ID
    metric_result = db.table("intimacy_metrics").select("id").eq(
        "story_id", str(story_id)
    ).eq("entity_id", str(entity_id)).execute()

    if not metric_result.data:
        return IntimacyHistoryResponse(
            characterId=str(entity_id),
            characterName=character_name,
            history=[],
        )

    metric_id = metric_result.data[0]["id"]

    # Get history
    history_result = db.table("intimacy_metrics_history").select("*").eq(
        "intimacy_metric_id", metric_id
    ).order("changed_at", desc=True).limit(limit).execute()

    history = [
        IntimacyHistoryEntry(
            metricName=h["metric_name"],
            oldValue=h.get("old_value", 50),
            newValue=h["new_value"],
            changeAmount=h.get("change_amount", 0),
            reason=h.get("reason"),
            changedAt=h["changed_at"],
        )
        for h in history_result.data
    ]

    return IntimacyHistoryResponse(
        characterId=str(entity_id),
        characterName=character_name,
        history=history,
    )


# ============================================
# PHYSICAL STATE
# ============================================


@router.get(
    "/stories/{story_id}/physical-states",
    response_model=PhysicalStateListResponse
)
async def get_physical_states(story_id: UUID) -> PhysicalStateListResponse:
    """
    Get current physical states for all tracked characters in a story.
    """
    from schemas.api import FrontendPhysicalState, PhysicalStateListResponse

    db = get_supabase_client()

    result = db.table("character_physical_states").select("*").eq(
        "story_id", str(story_id)
    ).eq("is_current", True).execute()

    states = [
        FrontendPhysicalState(
            characterName=s["character_name"],
            clothing=s.get("clothing") or [],
            position=s.get("position"),
            locationInScene=s.get("location_in_scene"),
            physicalContact=s.get("physical_contact") or [],
            temporaryStates=s.get("temporary_states") or [],
        )
        for s in result.data
    ]

    return PhysicalStateListResponse(states=states)


# ============================================
# SCENE STATE
# ============================================


@router.get("/stories/{story_id}/scene-state")
async def get_scene_state(story_id: UUID):
    """
    Get the current active scene state for a story.
    """
    from schemas.api import FrontendSceneState

    db = get_supabase_client()

    result = db.table("scene_states").select("*").eq(
        "story_id", str(story_id)
    ).eq("scene_active", True).order("started_at", desc=True).limit(1).execute()

    if not result.data:
        return None

    s = result.data[0]
    return FrontendSceneState(
        sceneType=s["scene_type"],
        sceneActive=s["scene_active"],
        participants=s.get("participants") or [],
        mood=s.get("mood"),
        interrupted=s.get("interrupted", False),
        consentEstablished=s.get("consent_established") or [],
    )


# ============================================
# HIERARCHICAL SUMMARIES
# ============================================


@router.get(
    "/stories/{story_id}/summaries/scenes",
    response_model=SceneSummaryListResponse
)
async def get_scene_summaries(
    story_id: UUID,
    session_id: Optional[UUID] = None,
    limit: int = Query(default=20, ge=1, le=100),
) -> SceneSummaryListResponse:
    """
    Get scene summaries for a story.

    Scene summaries are created every ~30 messages and capture key events.
    """
    db = get_supabase_client()

    query = db.table("scene_summaries").select("*").eq(
        "story_id", str(story_id)
    )

    if session_id:
        query = query.eq("session_id", str(session_id))

    result = query.order("created_at", desc=True).limit(limit).execute()

    # Get total count
    count_result = db.table("scene_summaries").select(
        "id", count="exact"
    ).eq("story_id", str(story_id)).execute()
    total = count_result.count or 0

    scenes = [
        FrontendSceneSummary(
            id=str(s["id"]),
            sceneNumber=s["scene_number"],
            summary=s["summary"],
            charactersPresent=s.get("characters_present") or [],
            keyEvents=s.get("key_events") or [],
            mood=s.get("mood"),
            messageStart=s.get("message_start"),
            messageEnd=s.get("message_end"),
            createdAt=s["created_at"],
        )
        for s in result.data
    ]

    return SceneSummaryListResponse(scenes=scenes, total=total)


@router.get(
    "/stories/{story_id}/summaries/chapters",
    response_model=ChapterSummaryListResponse
)
async def get_chapter_summaries(
    story_id: UUID,
    limit: int = Query(default=20, ge=1, le=100),
) -> ChapterSummaryListResponse:
    """
    Get chapter summaries for a story.

    Chapters are created per session and consolidate scene summaries.
    """
    db = get_supabase_client()

    result = db.table("chapter_summaries").select("*").eq(
        "story_id", str(story_id)
    ).order("chapter_number", desc=True).limit(limit).execute()

    # Get total count
    count_result = db.table("chapter_summaries").select(
        "id", count="exact"
    ).eq("story_id", str(story_id)).execute()
    total = count_result.count or 0

    chapters = [
        FrontendChapterSummary(
            id=str(c["id"]),
            chapterNumber=c["chapter_number"],
            title=c.get("title"),
            summary=c["summary"],
            majorEvents=c.get("major_events") or [],
            characterDevelopments=c.get("character_developments") or [],
            relationshipChanges=c.get("relationship_changes") or [],
            startSession=c["start_session"],
            endSession=c["end_session"],
            createdAt=c["created_at"],
        )
        for c in result.data
    ]

    return ChapterSummaryListResponse(chapters=chapters, total=total)


@router.get(
    "/stories/{story_id}/summaries/arcs",
    response_model=ArcSummaryListResponse
)
async def get_arc_summaries(
    story_id: UUID,
    limit: int = Query(default=10, ge=1, le=50),
) -> ArcSummaryListResponse:
    """
    Get story arc summaries.

    Arcs are created after ~5 chapters and capture major plot threads.
    """
    db = get_supabase_client()

    result = db.table("arc_summaries").select("*").eq(
        "story_id", str(story_id)
    ).order("arc_number", desc=True).limit(limit).execute()

    # Get total count
    count_result = db.table("arc_summaries").select(
        "id", count="exact"
    ).eq("story_id", str(story_id)).execute()
    total = count_result.count or 0

    arcs = [
        FrontendArcSummary(
            id=str(a["id"]),
            arcNumber=a["arc_number"],
            summary=a["summary"],
            majorEvents=a.get("major_events") or [],
            majorDecisions=a.get("major_decisions") or [],
            characterDevelopments=a.get("character_developments") or [],
            startSession=a["start_session"],
            endSession=a["end_session"],
            createdAt=a["created_at"],
        )
        for a in result.data
    ]

    return ArcSummaryListResponse(arcs=arcs, total=total)


@router.get(
    "/stories/{story_id}/summaries/context",
    response_model=HierarchicalContextResponse
)
async def get_hierarchical_context(
    story_id: UUID,
    max_scenes: int = Query(default=5, ge=1, le=20),
    max_chapters: int = Query(default=3, ge=1, le=10),
    max_arcs: int = Query(default=2, ge=1, le=5),
) -> HierarchicalContextResponse:
    """
    Get hierarchical summary context for building LLM prompts.

    Returns recent scenes, chapters, and arcs in a format ready for context building.
    """
    from api.dependencies import get_summarization_service

    summarization = get_summarization_service()

    context = summarization.get_hierarchical_context(
        story_id=story_id,
        max_scenes=max_scenes,
        max_chapters=max_chapters,
        max_arcs=max_arcs,
    )

    # Transform to frontend format
    scenes = [
        FrontendSceneSummary(
            id=str(s["id"]),
            sceneNumber=s["scene_number"],
            summary=s["summary"],
            charactersPresent=s.get("characters_present") or [],
            keyEvents=s.get("key_events") or [],
            mood=s.get("mood"),
            messageStart=s.get("message_start"),
            messageEnd=s.get("message_end"),
            createdAt=s["created_at"],
        )
        for s in context.get("recent_scenes", [])
    ]

    chapters = [
        FrontendChapterSummary(
            id=str(c["id"]),
            chapterNumber=c["chapter_number"],
            title=c.get("title"),
            summary=c["summary"],
            majorEvents=c.get("major_events") or [],
            characterDevelopments=c.get("character_developments") or [],
            relationshipChanges=c.get("relationship_changes") or [],
            startSession=c["start_session"],
            endSession=c["end_session"],
            createdAt=c["created_at"],
        )
        for c in context.get("recent_chapters", [])
    ]

    arcs = [
        FrontendArcSummary(
            id=str(a["id"]),
            arcNumber=a["arc_number"],
            summary=a["summary"],
            majorEvents=a.get("major_events") or [],
            majorDecisions=a.get("major_decisions") or [],
            characterDevelopments=a.get("character_developments") or [],
            startSession=a["start_session"],
            endSession=a["end_session"],
            createdAt=a["created_at"],
        )
        for a in context.get("story_arcs", [])
    ]

    return HierarchicalContextResponse(
        recentScenes=scenes,
        recentChapters=chapters,
        storyArcs=arcs,
    )


@router.post("/stories/{story_id}/summaries/generate-chapter")
async def generate_chapter_summary(
    story_id: UUID,
    session_id: UUID,
    session_number: int,
):
    """
    Manually trigger chapter summary generation for a session.

    This is typically called automatically when a session completes,
    but can be triggered manually for regeneration.
    """
    from api.dependencies import get_summarization_service

    summarization = get_summarization_service()

    try:
        chapter = summarization.create_session_chapter(
            story_id=story_id,
            session_id=session_id,
            session_number=session_number,
        )

        return {
            "success": True,
            "chapter": FrontendChapterSummary(
                id=str(chapter["id"]),
                chapterNumber=chapter["chapter_number"],
                title=chapter.get("title"),
                summary=chapter["summary"],
                majorEvents=chapter.get("major_events") or [],
                characterDevelopments=chapter.get("character_developments") or [],
                relationshipChanges=chapter.get("relationship_changes") or [],
                startSession=chapter["start_session"],
                endSession=chapter["end_session"],
                createdAt=chapter["created_at"],
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stories/{story_id}/summaries/generate-arc")
async def generate_arc_summary(
    story_id: UUID,
    start_session: int,
    end_session: int,
):
    """
    Manually trigger arc summary generation.

    Arcs consolidate multiple chapters into higher-level summaries.
    """
    from api.dependencies import get_summarization_service

    summarization = get_summarization_service()

    # Get next arc number
    arc_number = summarization.get_next_arc_number(story_id)

    try:
        arc = summarization.create_arc_summary(
            story_id=story_id,
            arc_number=arc_number,
            start_session=start_session,
            end_session=end_session,
        )

        return {
            "success": True,
            "arc": FrontendArcSummary(
                id=str(arc["id"]),
                arcNumber=arc["arc_number"],
                summary=arc["summary"],
                majorEvents=arc.get("major_events") or [],
                majorDecisions=arc.get("major_decisions") or [],
                characterDevelopments=arc.get("character_developments") or [],
                startSession=arc["start_session"],
                endSession=arc["end_session"],
                createdAt=arc["created_at"],
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# SEMANTIC MEMORY RETRIEVAL
# ============================================


class RelevantMemoriesRequest(BaseModel):
    """Request for relevant semantic memories."""

    charactersPresent: Optional[list[str]] = None
    currentEmotion: Optional[str] = None
    currentTopics: Optional[list[str]] = None
    maxMemories: int = 20


class ScoredMemory(BaseModel):
    """Semantic memory with relevance score."""

    id: str
    memoryText: str
    charactersInvolved: list[str]
    primaryEmotion: str
    topics: list[str]
    importance: float
    pinned: bool
    setupForPayoff: bool
    relevanceScore: Optional[float] = None
    createdAt: str


class RelevantMemoriesResponse(BaseModel):
    """Response with scored memories."""

    memories: list[ScoredMemory]
    total: int


@router.post(
    "/stories/{story_id}/memories/relevant",
    response_model=RelevantMemoriesResponse
)
async def get_relevant_memories(
    story_id: UUID,
    request: RelevantMemoriesRequest,
) -> RelevantMemoriesResponse:
    """
    Get relevant semantic memories with relevance scoring.

    Uses the SpicyChat Semantic Memory 2.0 pattern:
    - Pinned memories always included
    - Scored by recency, importance, character overlap, emotion/topic match
    - Returns top N most relevant memories

    This endpoint is designed for context building during chat.
    """
    from api.dependencies import get_context_service

    context = get_context_service()

    memories = await context.get_semantic_memories(
        story_id=story_id,
        characters_present=request.charactersPresent,
        current_emotion=request.currentEmotion,
        current_topics=request.currentTopics,
        max_memories=request.maxMemories,
    )

    scored_memories = [
        ScoredMemory(
            id=str(m["id"]),
            memoryText=m["memory_text"],
            charactersInvolved=m.get("characters_involved") or [],
            primaryEmotion=m.get("primary_emotion", ""),
            topics=m.get("topics") or [],
            importance=m.get("importance", 0.5),
            pinned=m.get("pinned", False),
            setupForPayoff=m.get("setup_for_payoff", False),
            relevanceScore=m.get("_relevance_score"),
            createdAt=m.get("created_at", ""),
        )
        for m in memories
    ]

    return RelevantMemoriesResponse(
        memories=scored_memories,
        total=len(scored_memories),
    )


@router.get("/stories/{story_id}/memories/formatted-context")
async def get_formatted_memory_context(
    story_id: UUID,
    characters: Optional[str] = None,
    emotion: Optional[str] = None,
    max_memories: int = Query(default=15, ge=1, le=50),
    include_scores: bool = False,
):
    """
    Get formatted semantic memories ready for LLM context injection.

    Returns a formatted string of the most relevant memories.
    Useful for building chat prompts.
    """
    from api.dependencies import get_context_service

    context = get_context_service()

    # Parse characters from comma-separated string
    chars_list = None
    if characters:
        chars_list = [c.strip() for c in characters.split(",") if c.strip()]

    memories = await context.get_semantic_memories(
        story_id=story_id,
        characters_present=chars_list,
        current_emotion=emotion,
        max_memories=max_memories,
    )

    formatted = context.format_memories_for_context(
        memories=memories,
        include_scores=include_scores,
    )

    return {
        "context": formatted,
        "memoryCount": len(memories),
    }