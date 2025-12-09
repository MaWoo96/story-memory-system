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


# ============================================
# FRONTEND-COMPATIBLE MODELS (story-ui)
# ============================================


class ContentBoundaries(BaseModel):
    """Content boundaries from world building wizard."""

    violenceLevel: Optional[str] = None  # "none" | "mild" | "moderate" | "graphic"
    romanceLevel: Optional[str] = None  # "none" | "fade-to-black" | "suggestive" | "explicit"
    languageLevel: Optional[str] = None  # "clean" | "mild" | "strong"


class WorldSetting(BaseModel):
    """World setting from onboarding wizard."""

    worldType: Optional[str] = None  # "fantasy" | "scifi" | "modern" | "historical" | "other"
    settingDescription: Optional[str] = None
    themes: Optional[list[str]] = None
    protagonistType: Optional[str] = None  # "player-insert" | "named-character"
    protagonistDetails: Optional[str] = None
    narrativeStyle: Optional[str] = None  # "second-person" | "third-person"
    pacing: Optional[str] = None  # "slow-burn" | "balanced" | "fast-paced"
    contentBoundaries: Optional[ContentBoundaries] = None
    keyLocations: Optional[list[str]] = None
    importantLore: Optional[str] = None
    onboardingCompleted: Optional[bool] = None


class FrontendStory(BaseModel):
    """Story in frontend-expected format."""

    id: str
    title: str
    premise: Optional[str] = None
    description: Optional[str] = None
    genre: Optional[str] = None
    tone: Optional[str] = None
    tags: Optional[list[str]] = None
    grokInstructions: Optional[str] = None
    coverImageUrl: Optional[str] = None
    status: Optional[str] = None
    isNsfw: Optional[bool] = None
    worldSetting: Optional[WorldSetting] = None
    createdAt: str
    lastPlayed: str


class FrontendStoryListResponse(BaseModel):
    """Story list response for frontend."""

    stories: list[FrontendStory]


class UpdateStoryRequest(BaseModel):
    """Request to update story settings."""

    title: Optional[str] = None
    premise: Optional[str] = None
    description: Optional[str] = None
    genre: Optional[str] = None
    tone: Optional[str] = None
    tags: Optional[list[str]] = None
    grokInstructions: Optional[str] = None
    coverImageUrl: Optional[str] = None
    status: Optional[str] = None
    isNsfw: Optional[bool] = None
    worldSetting: Optional[WorldSetting] = None


class RelationshipMeter(BaseModel):
    """Relationship meter for frontend."""

    type: str  # "trust" | "affection" | "fear"
    value: int
    label: str


class CharacterRelationship(BaseModel):
    """Character relationship for frontend."""

    characterId: str
    characterName: str
    description: str


class CharacterEvent(BaseModel):
    """Character event history for frontend."""

    timestamp: str
    event: str


class FrontendCharacter(BaseModel):
    """Character in frontend-expected format."""

    id: str
    name: str
    type: str  # "main" | "side" | "location"
    portraitUrl: Optional[str] = None
    importance: int  # 1-5
    relationshipMeter: Optional[RelationshipMeter] = None
    description: Optional[str] = None
    facts: Optional[list[str]] = None
    relationships: Optional[list[CharacterRelationship]] = None
    eventHistory: Optional[list[CharacterEvent]] = None


class FrontendCharacterListResponse(BaseModel):
    """Character list response for frontend."""

    characters: list[FrontendCharacter]


class FrontendCharacterDetailsResponse(BaseModel):
    """Character details response for frontend."""

    character: FrontendCharacter


class FrontendChatMessage(BaseModel):
    """Chat message in frontend-expected format."""

    id: str
    type: str  # "narrator" | "character" | "player-choice" | "system"
    content: str
    characterId: Optional[str] = None
    characterName: Optional[str] = None
    characterPortraitUrl: Optional[str] = None
    timestamp: str


class FrontendSessionContextResponse(BaseModel):
    """Session context response for frontend."""

    messages: list[FrontendChatMessage]


class FrontendCompleteSessionResponse(BaseModel):
    """Complete session response for frontend."""

    success: bool
    memoriesExtracted: int


class FrontendStat(BaseModel):
    """Protagonist stat for frontend."""

    name: str
    current: int
    max: int


class FrontendSkill(BaseModel):
    """Protagonist skill for frontend."""

    name: str
    rank: str  # "E" | "D" | "C" | "B" | "A" | "S"


class FrontendInventoryItem(BaseModel):
    """Inventory item for frontend."""

    id: str
    name: str
    iconUrl: Optional[str] = None
    description: str


class FrontendNPCRelationship(BaseModel):
    """NPC relationship for frontend."""

    id: str
    name: str
    portraitUrl: Optional[str] = None
    relationshipMeter: RelationshipMeter


class FrontendQuest(BaseModel):
    """Quest for frontend."""

    id: str
    description: str
    completed: bool


class FrontendLocation(BaseModel):
    """Location for frontend."""

    name: str
    imageUrl: Optional[str] = None


class FrontendWorldState(BaseModel):
    """World state for frontend."""

    location: FrontendLocation
    time: str
    quests: list[FrontendQuest]


class FrontendProtagonist(BaseModel):
    """Protagonist state for frontend."""

    stats: list[FrontendStat]
    skills: list[FrontendSkill]
    inventory: list[FrontendInventoryItem]
    statusEffects: list[str]


class FrontendGameState(BaseModel):
    """Game state for frontend."""

    protagonist: FrontendProtagonist
    npcRelationships: list[FrontendNPCRelationship]
    worldState: FrontendWorldState


class FrontendGameStateResponse(BaseModel):
    """Game state response for frontend."""

    gameState: FrontendGameState


class SelectedLoRA(BaseModel):
    """A LoRA with weight for generation."""
    name: str
    weight: float = Field(default=0.7, ge=0.0, le=1.5)


class GeneratePortraitRequest(BaseModel):
    """Request to generate portrait with full control over generation parameters."""

    style: str = Field(default="anime", description="Style: anime, realistic, or fantasy")
    seed: Optional[int] = None
    pose: Optional[str] = Field(default=None, description="Specific pose for the portrait")
    expression: Optional[str] = Field(default=None, description="Facial expression")
    additionalTags: Optional[list[str]] = Field(default=None, description="Extra prompt tags")
    # Extended parameters for full quality control
    custom_loras: Optional[list[SelectedLoRA]] = Field(default=None, description="Custom LoRAs to apply")
    use_standard_lora: Optional[bool] = Field(default=None, description="Use standard anime LoRA")
    use_default_loras: Optional[bool] = Field(default=True, description="Use auto-detected default LoRAs")
    width: int = Field(default=832, ge=256, le=1920, description="Image width")
    height: int = Field(default=1216, ge=256, le=1920, description="Image height")
    steps: int = Field(default=25, ge=10, le=50, description="Generation steps")
    cfg: float = Field(default=7.0, ge=1.0, le=20.0, description="CFG scale")
    hires_scale: float = Field(default=1.5, ge=1.0, le=2.0, description="HiRes upscale factor")
    hires_denoise: float = Field(default=0.55, ge=0.0, le=1.0, description="HiRes denoising strength")


class GeneratePortraitResponse(BaseModel):
    """Portrait generation response for frontend."""

    portraitUrl: str
    prompt: Optional[str] = None
    seed: Optional[int] = None
    model: Optional[str] = None
    lorasApplied: Optional[list[SelectedLoRA]] = Field(default=None, description="LoRAs that were applied")


# ============================================
# MEMORY MANAGEMENT (User-editable memories)
# ============================================


class FrontendSemanticMemory(BaseModel):
    """Semantic memory in frontend format."""

    id: str
    memoryText: str
    charactersInvolved: list[str]
    primaryEmotion: str
    topics: list[str]
    importance: float
    pinned: bool
    userEdited: bool
    hidden: bool
    setupForPayoff: bool
    createdAt: str
    sessionId: Optional[str] = None


class SemanticMemoryListResponse(BaseModel):
    """List of semantic memories."""

    memories: list[FrontendSemanticMemory]
    total: int


class UpdateMemoryRequest(BaseModel):
    """Request to update a memory."""

    memoryText: Optional[str] = None
    pinned: Optional[bool] = None
    hidden: Optional[bool] = None
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CreateMemoryRequest(BaseModel):
    """Request to manually create a memory."""

    memoryText: str
    charactersInvolved: list[str] = []
    primaryEmotion: str = "trust"
    topics: list[str] = []
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    pinned: bool = False


# ============================================
# INTIMACY METRICS (Multi-dimensional relationships)
# ============================================


class FrontendIntimacyMetrics(BaseModel):
    """Intimacy metrics for a character."""

    characterId: str
    characterName: str
    affection: int
    trust: int
    lust: int
    comfort: int
    jealousy: int
    submission: Optional[int] = None
    dominance: Optional[int] = None
    portraitUrl: Optional[str] = None


class IntimacyMetricsListResponse(BaseModel):
    """List of intimacy metrics."""

    metrics: list[FrontendIntimacyMetrics]


class IntimacyHistoryEntry(BaseModel):
    """Single entry in intimacy history."""

    metricName: str
    oldValue: int
    newValue: int
    changeAmount: int
    reason: Optional[str]
    changedAt: str


class IntimacyHistoryResponse(BaseModel):
    """History of intimacy changes for a character."""

    characterId: str
    characterName: str
    history: list[IntimacyHistoryEntry]


# ============================================
# PHYSICAL STATE
# ============================================


class FrontendPhysicalState(BaseModel):
    """Physical state for a character."""

    characterName: str
    clothing: list[str]
    position: Optional[str] = None
    locationInScene: Optional[str] = None
    physicalContact: list[str] = []
    temporaryStates: list[str] = []


class PhysicalStateListResponse(BaseModel):
    """Current physical states for a story."""

    states: list[FrontendPhysicalState]


# ============================================
# SCENE STATE
# ============================================


class FrontendSceneState(BaseModel):
    """Current scene state."""

    sceneType: str
    sceneActive: bool
    participants: list[str]
    mood: Optional[str] = None
    interrupted: bool = False
    consentEstablished: list[str] = []


# ============================================
# HIERARCHICAL SUMMARIES
# ============================================


class FrontendSceneSummary(BaseModel):
    """Scene summary in frontend format."""

    id: str
    sceneNumber: int
    summary: str
    charactersPresent: list[str]
    keyEvents: list[str]
    mood: Optional[str] = None
    messageStart: Optional[int] = None
    messageEnd: Optional[int] = None
    createdAt: str


class FrontendChapterSummary(BaseModel):
    """Chapter summary in frontend format."""

    id: str
    chapterNumber: int
    title: Optional[str] = None
    summary: str
    majorEvents: list[str]
    characterDevelopments: list[str]
    relationshipChanges: list[str]
    startSession: int
    endSession: int
    createdAt: str


class FrontendArcSummary(BaseModel):
    """Story arc summary in frontend format."""

    id: str
    arcNumber: int
    summary: str
    majorEvents: list[str]
    majorDecisions: list[str]
    characterDevelopments: list[str]
    startSession: int
    endSession: int
    createdAt: str


class SceneSummaryListResponse(BaseModel):
    """List of scene summaries."""

    scenes: list[FrontendSceneSummary]
    total: int


class ChapterSummaryListResponse(BaseModel):
    """List of chapter summaries."""

    chapters: list[FrontendChapterSummary]
    total: int


class ArcSummaryListResponse(BaseModel):
    """List of arc summaries."""

    arcs: list[FrontendArcSummary]
    total: int


class HierarchicalContextResponse(BaseModel):
    """Hierarchical summary context for building LLM prompts."""

    recentScenes: list[FrontendSceneSummary]
    recentChapters: list[FrontendChapterSummary]
    storyArcs: list[FrontendArcSummary]
