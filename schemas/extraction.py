"""
Pydantic schemas for memory extraction from story session transcripts.

These schemas are used with Grok 4.1 Fast structured outputs to guarantee
schema compliance during the extraction process.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ============================================
# ENUMS
# ============================================


class EntityType(str, Enum):
    CHARACTER = "character"
    LOCATION = "location"
    ITEM = "item"
    FACTION = "faction"
    CONCEPT = "concept"
    CREATURE = "creature"
    ORGANIZATION = "organization"


class RelationshipType(str, Enum):
    KNOWS = "knows"
    LOVES = "loves"
    HATES = "hates"
    FEARS = "fears"
    SERVES = "serves"
    OWNS = "owns"
    LOCATED_AT = "located_at"
    MEMBER_OF = "member_of"
    ALLIES = "allies"
    ENEMIES = "enemies"
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    SIBLING = "sibling"
    MARRIED_TO = "married_to"
    EMPLOYS = "employs"
    EMPLOYED_BY = "employed_by"
    MENTOR = "mentor"
    STUDENT_OF = "student_of"
    ROMANTIC_INTEREST = "romantic_interest"
    RIVALS = "rivals"
    BETRAYED = "betrayed"
    ALLIED_WITH = "allied_with"
    WITNESS_AGAINST = "witness_against"
    BELONGED_TO = "belonged_to"


class EventType(str, Enum):
    PLOT_POINT = "plot_point"
    REVELATION = "revelation"
    CONFLICT = "conflict"
    RESOLUTION = "resolution"
    INTRODUCTION = "introduction"
    DEPARTURE = "departure"
    TRANSFORMATION = "transformation"
    TRAINING = "training"
    GIFT = "gift"
    CONFESSION = "confession"
    THREAT = "threat"
    NEGOTIATION = "negotiation"
    CONFRONTATION = "confrontation"
    MAGICAL_INCIDENT = "magical_incident"
    OFFSCREEN = "offscreen"


class FactType(str, Enum):
    TRAIT = "trait"
    OCCUPATION = "occupation"
    POSSESSION = "possession"
    ABILITY = "ability"
    GOAL = "goal"
    SECRET = "secret"
    HISTORY = "history"
    KNOWLEDGE = "knowledge"
    MOTIVATION = "motivation"
    MECHANIC = "mechanic"
    EFFECT = "effect"
    FEATURE = "feature"
    ATMOSPHERE = "atmosphere"
    LOCATION = "location"


class StatType(str, Enum):
    PERCENTAGE = "percentage"
    RANK = "rank"
    INTEGER = "integer"
    STATUS = "status"


class InventoryAction(str, Enum):
    RECEIVED = "received"
    EQUIPPED = "equipped"
    WEARING = "wearing"
    LOST = "lost"
    CONSUMED = "consumed"
    GIVEN_AWAY = "given_away"
    DROPPED = "dropped"


class DecisionType(str, Enum):
    MADE = "made"
    PENDING = "pending"


class CharacterStatType(str, Enum):
    AFFECTION = "affection"
    TRUST = "trust"
    LOYALTY = "loyalty"
    FEAR = "fear"
    RESPECT = "respect"
    RIVALRY = "rivalry"


# ============================================
# NESTED MODELS
# ============================================


class Fact(BaseModel):
    fact_type: FactType = Field(description="Category of the fact")
    fact_value: str = Field(description="The actual fact content")
    importance: float = Field(description="Importance score 0.0-1.0", ge=0.0, le=1.0)


class Entity(BaseModel):
    entity_type: EntityType = Field(description="Type of entity")
    canonical_name: str = Field(description="Primary name for this entity")
    aliases: list[str] = Field(description="Other names/references used", default_factory=list)
    description: str = Field(description="Current description based on session")
    facts: list[Fact] = Field(description="Known facts about this entity", default_factory=list)
    importance: float = Field(description="Narrative importance 0.0-1.0", ge=0.0, le=1.0)


class Relationship(BaseModel):
    source: str = Field(description="Entity name or ID where relationship originates")
    target: str = Field(description="Entity name or ID that relationship points to")
    relationship_type: RelationshipType = Field(description="Type of relationship")
    description: str = Field(description="Nature of the relationship")
    importance: float = Field(description="Narrative importance 0.0-1.0", ge=0.0, le=1.0)


class Event(BaseModel):
    event_type: EventType = Field(description="Category of event")
    description: str = Field(description="What happened")
    participants: list[str] = Field(description="Entity names involved", default_factory=list)
    importance: float = Field(description="Narrative importance 0.0-1.0", ge=0.0, le=1.0)


class Decision(BaseModel):
    situation: str = Field(description="What prompted the choice")
    choice_made: Optional[str] = Field(
        description="What the protagonist chose (null if pending)", default=None
    )
    options: Optional[list[str]] = Field(
        description="Available options if pending", default=None
    )
    decision_type: DecisionType = Field(description="Whether decision was made or is pending")
    importance: float = Field(description="Narrative importance 0.0-1.0", ge=0.0, le=1.0)


class ProtagonistStat(BaseModel):
    stat: str = Field(description="Name of the stat")
    value: Optional[float | int | str] = Field(description="Current value", default=None)
    max: Optional[float | int] = Field(description="Maximum value if applicable", default=None)
    type: StatType = Field(description="Type of stat")
    change: Optional[str | float | int] = Field(description="Change this session", default=None)


class ProtagonistSkill(BaseModel):
    name: str = Field(description="Skill name")
    rank: Optional[str] = Field(description="Skill rank (E, D, C, B, A, S, etc.)", default=None)
    description: str = Field(description="What the skill does")
    mechanical_effect: Optional[str] = Field(description="Game mechanics", default=None)
    requirements: Optional[str] = Field(description="Requirements to use", default=None)
    cooldown: Optional[str] = Field(description="Cooldown if applicable", default=None)
    acquired_this_session: bool = Field(description="Whether learned this session")


class InventoryChange(BaseModel):
    action: InventoryAction = Field(description="What happened to the item")
    item: str = Field(description="Item name")
    description: Optional[str] = Field(description="Item description", default=None)
    properties: Optional[str] = Field(description="Mechanical properties", default=None)
    source: Optional[str] = Field(description="Where it came from", default=None)


class StatusEffect(BaseModel):
    effect: str = Field(description="Name of the effect")
    description: Optional[str] = Field(description="What it does", default=None)
    temporary: bool = Field(description="Whether effect is temporary")
    duration: Optional[str] = Field(description="How long it lasts", default=None)


class Protagonist(BaseModel):
    stats: list[ProtagonistStat] = Field(
        description="Current stat values", default_factory=list
    )
    skills: list[ProtagonistSkill] = Field(
        description="Skills and abilities", default_factory=list
    )
    inventory_changes: list[InventoryChange] = Field(
        description="Items gained/lost/used", default_factory=list
    )
    status_effects: list[StatusEffect] = Field(
        description="Active buffs/debuffs", default_factory=list
    )


class CharacterState(BaseModel):
    character: str = Field(description="Character name")
    stat_type: CharacterStatType = Field(description="Type of relationship stat")
    value: int = Field(description="Current value")
    max: int = Field(description="Maximum value")
    label: Optional[str] = Field(description="Descriptive label", default=None)
    change: Optional[str] = Field(description="Change this session (e.g., '+25')", default=None)


class WorldState(BaseModel):
    current_time: Optional[str] = Field(description="In-world time", default=None)
    current_location: Optional[str] = Field(description="Where protagonist is", default=None)
    pending_obligations: list[str] = Field(
        description="Things protagonist needs to do", default_factory=list
    )


class StoryExtraction(BaseModel):
    """Complete extraction from a story session."""

    entities: list[Entity] = Field(description="Characters, locations, items, etc.")
    relationships: list[Relationship] = Field(description="Connections between entities")
    events: list[Event] = Field(description="Things that happened")
    decisions: list[Decision] = Field(description="Choices made or pending")
    protagonist: Protagonist = Field(description="Player character state")
    character_states: list[CharacterState] = Field(description="NPC relationship meters")
    world_state: WorldState = Field(description="Current world/time state")
    session_summary: str = Field(description="2-3 paragraph summary")
    key_moments: list[str] = Field(description="3-7 most significant moments")
