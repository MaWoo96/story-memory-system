"""
Grok-powered real-time chat service for interactive storytelling.

Uses xAI Grok to generate narrative responses based on story context,
character knowledge, and user input.

Research-backed prompt patterns from:
- SpicyChat Semantic Memory 2.0 patterns
- SillyTavern lorebook/context injection
- NovelAI ATTG metadata format
- Emotional RAG and salience-based selection
"""

import os
from typing import Optional, AsyncGenerator
from pydantic import BaseModel, Field
from openai import OpenAI
from uuid import UUID
from datetime import datetime

from services.prompts import (
    build_narrator_prompt,
    build_context_block,
    build_post_history_instructions,
    build_immersive_opening_prompt,
    build_world_context_for_opening,
    build_protagonist_context,
    ATTG_HEADER,
)


class ChatMessage(BaseModel):
    """A single chat message."""
    id: str
    type: str = Field(description="narrator, character, player-choice, system, or image")
    content: str
    character_id: Optional[str] = None
    character_name: Optional[str] = None
    character_portrait_url: Optional[str] = None


class IntroducedNPC(BaseModel):
    """An NPC introduced or significantly featured in the narration."""
    name: str = Field(description="Character's full name")
    apparent_age: Optional[int] = Field(default=None, description="Apparent age if relevant")
    physical_description: str = Field(description="Vivid physical description: hair, eyes, body, notable features")
    clothing_description: Optional[str] = Field(default=None, description="What they're wearing")
    personality_hints: Optional[str] = Field(default=None, description="Brief personality indicator from behavior")
    role: Optional[str] = Field(default=None, description="Their role: guide, love_interest, antagonist, mentor, etc.")
    importance: float = Field(default=0.5, description="0.0-1.0 importance to the story")


class ProtagonistStatus(BaseModel):
    """Current status of the protagonist."""
    health: Optional[int] = Field(default=100, ge=0, le=100)
    stamina: Optional[int] = Field(default=100, ge=0, le=100)
    arousal: Optional[int] = Field(default=0, ge=0, le=100)
    stress: Optional[int] = Field(default=0, ge=0, le=100)
    hunger: Optional[int] = Field(default=0, ge=0, le=100)
    custom_stats: Optional[dict[str, int]] = Field(
        default=None,
        description="World-specific stats like mana, eros_mana, corruption, etc."
    )
    status_effects: Optional[list[str]] = Field(
        default=None,
        description="Active status effects like 'aroused', 'confused', 'blessed'"
    )


class WorldStateUpdate(BaseModel):
    """Changes to the world state from this narration."""
    current_location: Optional[str] = Field(default=None, description="Where the protagonist is now")
    time_of_day: Optional[str] = Field(default=None, description="dawn, morning, noon, afternoon, evening, night")
    discovered_facts: Optional[list[str]] = Field(
        default=None,
        description="New facts about the world the protagonist learned"
    )
    unlocked_locations: Optional[list[str]] = Field(
        default=None,
        description="New locations the protagonist can now travel to"
    )


class NarratorResponse(BaseModel):
    """Structured response from Grok for narration."""
    narration: str = Field(description="The narrator's description of the scene and events")
    character_dialogue: Optional[str] = Field(
        default=None,
        description="If a character speaks, their dialogue"
    )
    speaking_character: Optional[str] = Field(
        default=None,
        description="Name of the character speaking, if any"
    )
    suggested_choices: list[str] = Field(
        default_factory=list,
        description="2-4 suggested player actions/choices"
    )
    scene_image_suggestion: Optional[str] = Field(
        default=None,
        description="If the scene warrants an image, a brief description for generation"
    )
    mood: str = Field(
        default="neutral",
        description="Current scene mood: tense, romantic, comedic, action, peaceful, mysterious, intimate"
    )
    # Enhanced fields for immersive storytelling
    npcs_introduced: Optional[list[IntroducedNPC]] = Field(
        default=None,
        description="New NPCs introduced or significantly featured in this narration. Include vivid physical descriptions."
    )
    protagonist_status: Optional[ProtagonistStatus] = Field(
        default=None,
        description="Updated protagonist status after this scene (health, stamina, arousal, custom stats)"
    )
    world_state_update: Optional[WorldStateUpdate] = Field(
        default=None,
        description="Changes to the world state (location, time, discovered facts)"
    )
    asks_protagonist_info: Optional[str] = Field(
        default=None,
        description="If the narrator needs to know something about the protagonist (name, appearance, choice), the question to ask"
    )


class ChatService:
    """
    Real-time chat service using Grok for interactive storytelling.

    Maintains conversation context and generates appropriate narrative
    responses based on story premise, character knowledge, and player input.

    Uses research-backed prompt patterns:
    - Implicit memory surfacing through character voice
    - Post-history instructions for stronger influence (NovelAI pattern)
    - ATTG metadata format models recognize
    - Emotion-aware context building
    """

    def __init__(self):
        """Initialize the chat service with xAI Grok."""
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        self.model = "grok-4-1-fast-reasoning"

    def generate_response(
        self,
        player_message: str,
        story_premise: str,
        story_title: Optional[str] = None,
        story_genre: Optional[str] = None,
        story_tags: Optional[list[str]] = None,
        current_situation: Optional[dict] = None,
        characters_present: Optional[list[dict]] = None,
        recent_events: Optional[list[str]] = None,
        conversation_history: Optional[list[dict]] = None,
        relationship_meters: Optional[list[dict]] = None,
        physical_states: Optional[list[dict]] = None,
        semantic_memories: Optional[list[dict]] = None,
        scene_state: Optional[dict] = None,
        is_nsfw: bool = True,
        custom_instructions: Optional[str] = None,
    ) -> NarratorResponse:
        """
        Generate a narrative response to player input.

        Uses research-backed context building:
        - System prompt with NSFW/SFW-appropriate guidelines
        - ATTG metadata header for model recognition
        - Structured context block with all state
        - Post-history instructions near end for stronger influence
        - Token budget: ~40% recent, ~25% memories, ~15% system, ~15% state

        Args:
            player_message: What the player said/did
            story_premise: The story's basic premise
            story_title: Story title for ATTG header
            story_genre: Genre for ATTG header
            story_tags: Tags for ATTG header
            current_situation: Current scene info (location, time, description)
            characters_present: Characters in scene with descriptions
            recent_events: Recent narrative events for context
            conversation_history: Previous messages for context
            relationship_meters: Current relationship values per character
            physical_states: Physical states (clothing, position, etc.)
            semantic_memories: Retrieved semantic memories with scores
            scene_state: Current scene type and state
            is_nsfw: Whether NSFW content is allowed
            custom_instructions: Story-specific instructions for the AI

        Returns:
            NarratorResponse with narration and optional character dialogue
        """
        # Build system prompt using research-backed templates
        system_prompt = build_narrator_prompt(
            is_nsfw=is_nsfw,
            custom_instructions=custom_instructions,
        )

        # Add ATTG metadata header if story info available
        if story_title:
            attg_header = ATTG_HEADER.substitute(
                title=story_title,
                genre=story_genre or "Interactive Fiction",
                tags=", ".join(story_tags) if story_tags else "roleplay, interactive",
            )
            system_prompt = f"{attg_header}\n\n{system_prompt}"

        # Build context block using research-backed templates
        context = build_context_block(
            story_summary=story_premise,
            recent_events=recent_events,
            current_situation=current_situation,
            characters_present=characters_present,
            relationship_meters=relationship_meters,
            physical_states=physical_states,
            memories=semantic_memories,
            scene_state=scene_state,
        )

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context}"},
        ]

        # Add conversation history (last 20 messages for ~40% of context budget)
        if conversation_history:
            for msg in conversation_history[-20:]:
                role = "assistant" if msg.get("type") in ["narrator", "character"] else "user"
                messages.append({"role": role, "content": msg.get("content", "")})

        # Get character names for post-history instructions
        char_names = [c.get("name", "Unknown") for c in (characters_present or [])]
        current_mood = scene_state.get("mood") if scene_state else "neutral"

        # Add post-history instructions (NovelAI pattern - stronger influence near end)
        post_history = build_post_history_instructions(
            characters_present=char_names,
            current_mood=current_mood,
        )

        # Add current player message with post-history instructions
        messages.append({
            "role": "user",
            "content": f"{post_history}\n\nPLAYER ACTION: {player_message}\n\nGenerate the narrative response."
        })

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=NarratorResponse,
                temperature=0.8,  # Creative but not too random
            )

            return completion.choices[0].message.parsed

        except Exception as e:
            print(f"[ChatService] Grok API error: {e}")
            # Return a fallback response
            return NarratorResponse(
                narration="You consider your next move. The story continues...\n\n(System note: AI response unavailable)",
                suggested_choices=["Look around", "Wait", "Leave"],
                mood="neutral",
            )

    def generate_opening(
        self,
        story_premise: str,
        story_title: Optional[str] = None,
        story_genre: Optional[str] = None,
        story_tags: Optional[list[str]] = None,
        characters: Optional[list[dict]] = None,
        starting_location: Optional[str] = None,
        is_nsfw: bool = True,
        custom_instructions: Optional[str] = None,
        world_setting: Optional[dict] = None,
        protagonist_details: Optional[str] = None,
    ) -> NarratorResponse:
        """
        Generate an IMMERSIVE opening narration for a new story session.

        Uses the new immersive opening system that:
        - Grounds player through sensory experience, not exposition
        - Introduces NPCs with full physical descriptions
        - Discovers world organically through action
        - Asks for protagonist details naturally through characters

        Args:
            story_premise: The story's basic premise
            story_title: Story title for ATTG header
            story_genre: Genre for ATTG header
            story_tags: Tags for ATTG header
            characters: Main characters to potentially introduce
            starting_location: Where the story begins
            is_nsfw: Whether NSFW content is allowed
            custom_instructions: Story-specific instructions for the AI
            world_setting: Full world settings from wizard (worldType, themes, lore, etc.)
            protagonist_details: Details about the protagonist

        Returns:
            NarratorResponse with immersive opening narration, NPC introductions, etc.
        """
        # Build the immersive opening system prompt
        system_prompt = build_immersive_opening_prompt(
            world_setting=world_setting,
            custom_instructions=custom_instructions,
            is_nsfw=is_nsfw,
        )

        # Add ATTG metadata header if story info available
        if story_title:
            attg_header = ATTG_HEADER.substitute(
                title=story_title,
                genre=story_genre or "Interactive Fiction",
                tags=", ".join(story_tags) if story_tags else "roleplay, interactive",
            )
            system_prompt = f"{attg_header}\n\n{system_prompt}"

        # Build context for the opening
        context_parts = []

        # Story premise
        context_parts.append(f"STORY PREMISE:\n{story_premise}")

        # World setting context (detailed, for AI reference)
        world_context = build_world_context_for_opening(world_setting)
        if world_context:
            context_parts.append(world_context)

        # Protagonist context
        protagonist_context = build_protagonist_context(protagonist_details)
        context_parts.append(protagonist_context)

        # Characters that may appear (AI should introduce them with vivid descriptions)
        if characters:
            chars = characters[:5]  # More characters available for opening
            chars_str = "\n".join([
                f"• {c.get('name', 'Unknown')}: {c.get('description', 'No description')}"
                for c in chars
            ])
            context_parts.append(f"CHARACTERS AVAILABLE TO INTRODUCE (describe vividly when they appear):\n{chars_str}")

        # Starting location hint
        if starting_location:
            context_parts.append(f"STARTING LOCATION HINT: {starting_location}\n(Reveal through sensory details, don't name directly at first)")

        context = "\n\n".join(context_parts)

        # Build the final user message
        user_message = f"""{context}

---

NOW GENERATE THE IMMERSIVE OPENING:

Create a 3-5 paragraph opening that:
1. STARTS with the protagonist's immediate sensory experience (waking up, feeling disoriented, etc.)
2. Slowly reveals the environment through what they see, hear, smell, feel
3. Introduces at least one NPC with FULL physical description in the npcs_introduced field
4. Ends with a situation that demands a response (a character speaking, a choice to make)
5. ALWAYS suggests a scene image in scene_image_suggestion
6. If protagonist name is unknown, use asks_protagonist_info to have a character ask

DO NOT:
- Start with exposition about the world
- Explain the world's rules directly
- Have the protagonist already know what's happening
- Skip sensory grounding"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=NarratorResponse,
                temperature=0.85,  # Slightly higher for creative openings
            )

            return completion.choices[0].message.parsed

        except Exception as e:
            print(f"[ChatService] Grok API error for opening: {e}")
            import traceback
            traceback.print_exc()
            # Fallback with basic opening
            return NarratorResponse(
                narration=f"You wake with a start. Something is different—wrong, perhaps, or simply unfamiliar. The ground beneath you is soft, yielding. Your eyes flutter open to reveal... {starting_location or 'a place unlike any you remember'}.\n\n{story_premise}",
                suggested_choices=[
                    "Sit up slowly and take in your surroundings",
                    "Call out—is anyone there?",
                    "Check yourself for injuries",
                    "Do something else..."
                ],
                mood="mysterious",
                asks_protagonist_info="What is your name?",
                scene_image_suggestion=f"A person waking up in {starting_location or 'an unfamiliar place'}, disoriented, mysterious atmosphere",
            )


# Singleton instance
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get or create the chat service singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
