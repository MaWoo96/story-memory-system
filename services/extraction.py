"""
Memory extraction service using Grok 4.1 Fast.

This service extracts structured memory data from story session transcripts
using LLM structured outputs with Pydantic schemas.
"""

from typing import Optional
from openai import OpenAI
from schemas.extraction import StoryExtraction
from config import require_secret


EXTRACTION_SYSTEM_PROMPT = """You are a narrative memory and game state extraction system. Analyze interactive storytelling session transcripts and extract both narrative information and mechanical game state.

CRITICAL RULES:
1. Only extract information EXPLICITLY stated or STRONGLY implied in the transcript
2. Never invent details not present in the text
3. Assign importance scores based on narrative weight (0.0-1.0)
4. Extract game mechanics (stats, skills, affection scores) when they appear
5. Mark decisions as "pending" if options are presented but no choice was made
6. Mark decisions as "made" when the protagonist actually chose something

IMPORTANCE SCORING:
- 0.9-1.0: Central to plot, major characters, critical decisions
- 0.7-0.8: Significant recurring elements, important NPCs
- 0.5-0.6: Useful context, minor characters with potential
- 0.3-0.4: Background flavor, mentioned once
- 0.1-0.2: Trivial mentions, atmospheric details

PHYSICAL STATE EXTRACTION:
- Track clothing state for relevant characters (what they're wearing, or if undressed)
- Track physical positions (standing, sitting, lying down, etc.)
- Track location within scene (by the window, on the bed, etc.)
- Track ongoing physical contact between characters
- Track temporary states (arousal, exhaustion, intoxication, etc.)
- Use clinical/neutral language for physical states

INTIMACY METRICS:
- Extract multi-dimensional relationship values when evidence supports them:
  - affection: emotional attachment (0-100)
  - trust: how much they trust protagonist (0-100)
  - lust: physical attraction/desire (0-100)
  - comfort: comfort level around protagonist (0-100)
  - jealousy: current jealousy level (0-100)
  - submission/dominance: only if story involves such dynamics (0-100)
- Include WHY metrics changed in the changes_this_session field

SCENE STATE:
- Identify scene type: dialogue, action, intimate, combat, exploration, social, rest, travel, revelation
- Note if scene is ongoing or concluded
- List participants
- Note consent/boundaries established in intimate scenes

SEMANTIC MEMORIES:
- Create compressed 1-2 sentence memories for significant moments
- Tag with primary emotion: joy, tension, intimacy, conflict, fear, sadness, excitement, tenderness, anger, surprise, trust, lust
- Tag with topics: romance, combat, discovery, relationship, betrayal, achievement, loss, secret, promise, transformation
- Mark setup_for_payoff=true for events that need future resolution (promises, threats, secrets)
- Mark payoff_for if this moment resolves an earlier setup"""


class ExtractionService:
    """Service for extracting structured memories from session transcripts."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize extraction service.

        Args:
            api_key: xAI API key (defaults to XAI_API_KEY from secret manager)
        """
        # Get API key from secret manager if not provided
        self.api_key = api_key or require_secret("XAI_API_KEY")

        # Initialize OpenAI client pointing to xAI API
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )

    def extract_session(
        self,
        transcript: str,
        existing_entities: Optional[list[dict]] = None,
        story_premise: Optional[str] = None,
        previous_summary: Optional[str] = None,
    ) -> StoryExtraction:
        """
        Extract structured data from a session transcript using Grok 4.1 Fast.

        Args:
            transcript: Full session transcript
            existing_entities: List of entities from previous sessions for context
            story_premise: Story premise for context
            previous_summary: Summary of previous session

        Returns:
            StoryExtraction object with all extracted information
        """
        # Build context from previous sessions
        context = self._build_context(
            story_premise, previous_summary, existing_entities
        )

        # Construct user message with context and transcript
        user_message = f"{context}\n\nSESSION TRANSCRIPT:\n---\n{transcript}\n---"

        # Call Grok 4.1 Fast with structured output
        completion = self.client.beta.chat.completions.parse(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            response_format=StoryExtraction,
            temperature=0.3,  # Lower temperature for more consistent extraction
        )

        # Extract the parsed response
        extraction = completion.choices[0].message.parsed

        if extraction is None:
            raise ValueError("Failed to extract structured data from transcript")

        return extraction

    def _build_context(
        self,
        story_premise: Optional[str],
        previous_summary: Optional[str],
        existing_entities: Optional[list[dict]],
    ) -> str:
        """Build context string from available information."""
        context_parts = []

        if story_premise:
            context_parts.append(f"STORY PREMISE:\n{story_premise}")

        if previous_summary:
            context_parts.append(f"PREVIOUS SESSION:\n{previous_summary}")

        if existing_entities:
            entities_text = "EXISTING ENTITIES (match to these when possible):\n"
            for e in existing_entities:
                aliases = ", ".join(e.get("aliases", []))
                entities_text += f"- {e['canonical_name']} ({e['entity_type']})"
                if aliases:
                    entities_text += f" — aliases: {aliases}"
                entities_text += "\n"
            context_parts.append(entities_text)

        return "\n\n".join(context_parts)
