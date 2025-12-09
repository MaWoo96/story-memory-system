"""
Hierarchical summarization service.

Implements three levels of summarization:
1. Scene summaries (~30 messages or scene boundary)
2. Chapter summaries (session boundary, consolidates scenes)
3. Arc summaries (periodic, consolidates chapters)

Based on research from SillyTavern's rolling summary system.
"""

from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field
from openai import OpenAI
from config import require_secret


# ============================================
# PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT
# ============================================


class SceneSummaryOutput(BaseModel):
    """Output schema for scene summarization."""

    summary: str = Field(description="2-3 sentence summary of the scene")
    characters_present: list[str] = Field(
        description="Names of characters in this scene",
        default_factory=list
    )
    key_events: list[str] = Field(
        description="1-3 key events that happened",
        default_factory=list
    )
    mood: str = Field(description="Emotional tone of the scene")
    scene_type: str = Field(
        description="Type: dialogue, action, intimate, combat, exploration, social, rest, travel, revelation"
    )


class ChapterSummaryOutput(BaseModel):
    """Output schema for chapter summarization."""

    title: str = Field(description="Short evocative title for this chapter")
    summary: str = Field(description="3-5 sentence summary of the chapter")
    major_events: list[str] = Field(
        description="3-5 most significant events",
        default_factory=list
    )
    character_developments: list[str] = Field(
        description="Notable character changes or revelations",
        default_factory=list
    )
    relationship_changes: list[str] = Field(
        description="Relationship changes between characters",
        default_factory=list
    )


class ArcSummaryOutput(BaseModel):
    """Output schema for arc summarization."""

    summary: str = Field(description="5-7 sentence summary of this story arc")
    major_events: list[str] = Field(
        description="5-10 most significant events across the arc",
        default_factory=list
    )
    major_decisions: list[str] = Field(
        description="Important choices the protagonist made",
        default_factory=list
    )
    character_developments: list[str] = Field(
        description="How characters grew or changed",
        default_factory=list
    )


# ============================================
# SYSTEM PROMPTS
# ============================================


SCENE_SUMMARY_PROMPT = """You are a narrative summarization system. Create a concise scene summary.

RULES:
1. Focus on WHAT HAPPENED, not literary analysis
2. Include character names and actions
3. Capture the emotional tone
4. Note any important revelations or decisions
5. Be specific, not vague

OUTPUT FORMAT:
- summary: 2-3 sentences describing what happened
- characters_present: List of character names in this scene
- key_events: 1-3 most important things that happened
- mood: The emotional tone (tense, romantic, playful, etc.)
- scene_type: dialogue/action/intimate/combat/exploration/social/rest/travel/revelation"""


CHAPTER_SUMMARY_PROMPT = """You are a narrative summarization system. Create a chapter summary from multiple scene summaries.

RULES:
1. Synthesize scenes into a coherent narrative
2. Focus on plot advancement and character development
3. Note relationship changes
4. Capture the overall arc of the chapter
5. Create an evocative title that hints at the content

INPUT: You will receive scene summaries in order. Synthesize them.

OUTPUT FORMAT:
- title: Short, evocative title (3-6 words)
- summary: 3-5 sentences covering the chapter's events
- major_events: 3-5 most significant events
- character_developments: Notable character changes
- relationship_changes: How relationships evolved"""


ARC_SUMMARY_PROMPT = """You are a narrative summarization system. Create an arc summary from multiple chapter summaries.

RULES:
1. Synthesize chapters into a coherent story arc
2. Focus on major plot threads and their progression
3. Note significant character transformations
4. Capture key decisions and their consequences
5. This summary should capture the essence of this part of the story

INPUT: You will receive chapter summaries in order. Synthesize them into an arc summary.

OUTPUT FORMAT:
- summary: 5-7 sentences covering the arc's events and themes
- major_events: 5-10 most significant events across the arc
- major_decisions: Important choices the protagonist made
- character_developments: How characters grew or changed"""


# ============================================
# SERVICE CLASS
# ============================================


class SummarizationService:
    """Service for creating hierarchical summaries."""

    # Configuration
    MESSAGES_PER_SCENE = 30  # Approximate messages before scene summary
    SCENES_PER_CHAPTER = 5  # Scenes before chapter summary
    CHAPTERS_PER_ARC = 5  # Chapters before arc summary

    def __init__(self, db_client, api_key: Optional[str] = None):
        """
        Initialize summarization service.

        Args:
            db_client: Supabase client instance
            api_key: xAI API key (defaults to XAI_API_KEY from secret manager)
        """
        self.db = db_client
        self.api_key = api_key or require_secret("XAI_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )

    # ============================================
    # SCENE SUMMARIZATION
    # ============================================

    def create_scene_summary(
        self,
        story_id: UUID,
        session_id: UUID,
        messages: list[dict],
        scene_number: int,
        message_start: int,
        message_end: int,
    ) -> dict:
        """
        Create a summary for a scene (chunk of messages).

        Args:
            story_id: Story UUID
            session_id: Session UUID
            messages: List of message dicts with 'role' and 'content'
            scene_number: Scene number within the session
            message_start: Starting message index
            message_end: Ending message index

        Returns:
            Created scene summary record
        """
        # Format messages for summarization
        transcript = self._format_messages(messages)

        # Generate summary using Grok
        completion = self.client.beta.chat.completions.parse(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "system", "content": SCENE_SUMMARY_PROMPT},
                {"role": "user", "content": f"Summarize this scene:\n\n{transcript}"}
            ],
            response_format=SceneSummaryOutput,
            temperature=0.3,
        )

        result = completion.choices[0].message.parsed
        if result is None:
            raise ValueError("Failed to generate scene summary")

        # Store in database
        record = self.db.table("scene_summaries").insert({
            "story_id": str(story_id),
            "session_id": str(session_id),
            "scene_number": scene_number,
            "summary": result.summary,
            "characters_present": result.characters_present,
            "key_events": result.key_events,
            "mood": result.mood,
            "message_start": message_start,
            "message_end": message_end,
        }).execute()

        return record.data[0]

    def should_create_scene_summary(
        self,
        message_count: int,
        last_scene_end: int,
    ) -> bool:
        """Check if we should create a new scene summary."""
        messages_since_last = message_count - last_scene_end
        return messages_since_last >= self.MESSAGES_PER_SCENE

    # ============================================
    # CHAPTER SUMMARIZATION
    # ============================================

    def create_chapter_summary(
        self,
        story_id: UUID,
        chapter_number: int,
        start_session: int,
        end_session: int,
    ) -> dict:
        """
        Create a chapter summary from scene summaries.

        Args:
            story_id: Story UUID
            chapter_number: Chapter number
            start_session: Starting session number
            end_session: Ending session number

        Returns:
            Created chapter summary record
        """
        # Get scene summaries for this chapter
        scenes_result = self.db.table("scene_summaries").select(
            "summary, characters_present, key_events, mood"
        ).eq("story_id", str(story_id)).gte(
            "session_id",
            self._get_session_id_by_number(story_id, start_session)
        ).lte(
            "session_id",
            self._get_session_id_by_number(story_id, end_session)
        ).order("created_at").execute()

        scenes = scenes_result.data

        if not scenes:
            raise ValueError(f"No scene summaries found for chapter {chapter_number}")

        # Format scenes for summarization
        scenes_text = self._format_scenes_for_chapter(scenes)

        # Generate chapter summary
        completion = self.client.beta.chat.completions.parse(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "system", "content": CHAPTER_SUMMARY_PROMPT},
                {"role": "user", "content": f"Create a chapter summary from these scenes:\n\n{scenes_text}"}
            ],
            response_format=ChapterSummaryOutput,
            temperature=0.3,
        )

        result = completion.choices[0].message.parsed
        if result is None:
            raise ValueError("Failed to generate chapter summary")

        # Store in database (upsert to handle re-generation)
        record = self.db.table("chapter_summaries").upsert({
            "story_id": str(story_id),
            "chapter_number": chapter_number,
            "title": result.title,
            "summary": result.summary,
            "major_events": result.major_events,
            "character_developments": result.character_developments,
            "relationship_changes": result.relationship_changes,
            "start_session": start_session,
            "end_session": end_session,
        }, on_conflict="story_id,chapter_number").execute()

        return record.data[0]

    def create_session_chapter(
        self,
        story_id: UUID,
        session_id: UUID,
        session_number: int,
    ) -> dict:
        """
        Create a chapter summary for a completed session.

        Simplified version that creates one chapter per session.

        Args:
            story_id: Story UUID
            session_id: Session UUID
            session_number: Session number

        Returns:
            Created chapter summary record
        """
        # Get scene summaries for this session
        scenes_result = self.db.table("scene_summaries").select(
            "summary, characters_present, key_events, mood"
        ).eq("session_id", str(session_id)).order("scene_number").execute()

        scenes = scenes_result.data

        # If no scenes, create a chapter from the session summary
        if not scenes:
            session_result = self.db.table("sessions").select(
                "summary, key_moments"
            ).eq("id", str(session_id)).execute()

            if session_result.data and session_result.data[0].get("summary"):
                session = session_result.data[0]
                return self.db.table("chapter_summaries").upsert({
                    "story_id": str(story_id),
                    "chapter_number": session_number,
                    "title": f"Session {session_number}",
                    "summary": session["summary"],
                    "major_events": session.get("key_moments") or [],
                    "character_developments": [],
                    "relationship_changes": [],
                    "start_session": session_number,
                    "end_session": session_number,
                }, on_conflict="story_id,chapter_number").execute().data[0]
            else:
                raise ValueError(f"No scene summaries or session summary for session {session_id}")

        # Format scenes for summarization
        scenes_text = self._format_scenes_for_chapter(scenes)

        # Generate chapter summary
        completion = self.client.beta.chat.completions.parse(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "system", "content": CHAPTER_SUMMARY_PROMPT},
                {"role": "user", "content": f"Create a chapter summary from these scenes:\n\n{scenes_text}"}
            ],
            response_format=ChapterSummaryOutput,
            temperature=0.3,
        )

        result = completion.choices[0].message.parsed
        if result is None:
            raise ValueError("Failed to generate chapter summary")

        # Store in database
        record = self.db.table("chapter_summaries").upsert({
            "story_id": str(story_id),
            "chapter_number": session_number,
            "title": result.title,
            "summary": result.summary,
            "major_events": result.major_events,
            "character_developments": result.character_developments,
            "relationship_changes": result.relationship_changes,
            "start_session": session_number,
            "end_session": session_number,
        }, on_conflict="story_id,chapter_number").execute()

        return record.data[0]

    # ============================================
    # ARC SUMMARIZATION
    # ============================================

    def create_arc_summary(
        self,
        story_id: UUID,
        arc_number: int,
        start_session: int,
        end_session: int,
    ) -> dict:
        """
        Create an arc summary from chapter summaries.

        Args:
            story_id: Story UUID
            arc_number: Arc number
            start_session: Starting session number
            end_session: Ending session number

        Returns:
            Created arc summary record
        """
        # Get chapter summaries for this arc
        chapters_result = self.db.table("chapter_summaries").select(
            "title, summary, major_events, character_developments, relationship_changes"
        ).eq("story_id", str(story_id)).gte(
            "start_session", start_session
        ).lte("end_session", end_session).order("chapter_number").execute()

        chapters = chapters_result.data

        if not chapters:
            raise ValueError(f"No chapter summaries found for arc {arc_number}")

        # Format chapters for summarization
        chapters_text = self._format_chapters_for_arc(chapters)

        # Generate arc summary
        completion = self.client.beta.chat.completions.parse(
            model="grok-4-1-fast-reasoning",
            messages=[
                {"role": "system", "content": ARC_SUMMARY_PROMPT},
                {"role": "user", "content": f"Create an arc summary from these chapters:\n\n{chapters_text}"}
            ],
            response_format=ArcSummaryOutput,
            temperature=0.3,
        )

        result = completion.choices[0].message.parsed
        if result is None:
            raise ValueError("Failed to generate arc summary")

        # Store in database (upsert to handle re-generation)
        # Note: arc_summaries uses TEXT[] not JSONB
        record = self.db.table("arc_summaries").upsert({
            "story_id": str(story_id),
            "arc_number": arc_number,
            "start_session": start_session,
            "end_session": end_session,
            "summary": result.summary,
            "major_events": result.major_events,  # TEXT[]
            "major_decisions": result.major_decisions,  # TEXT[]
            "character_developments": result.character_developments,  # TEXT[]
        }, on_conflict="story_id,arc_number").execute()

        return record.data[0]

    def should_create_arc_summary(
        self,
        story_id: UUID,
        current_session: int,
    ) -> bool:
        """Check if we should create a new arc summary."""
        # Get latest arc end session
        result = self.db.table("arc_summaries").select(
            "end_session"
        ).eq("story_id", str(story_id)).order(
            "arc_number", desc=True
        ).limit(1).execute()

        if not result.data:
            # No arcs yet, create first arc after CHAPTERS_PER_ARC sessions
            return current_session >= self.CHAPTERS_PER_ARC
        else:
            last_arc_end = result.data[0]["end_session"]
            return current_session - last_arc_end >= self.CHAPTERS_PER_ARC

    def get_next_arc_number(self, story_id: UUID) -> int:
        """Get the next arc number for a story."""
        result = self.db.table("arc_summaries").select(
            "arc_number"
        ).eq("story_id", str(story_id)).order(
            "arc_number", desc=True
        ).limit(1).execute()

        if not result.data:
            return 1
        return result.data[0]["arc_number"] + 1

    # ============================================
    # HELPER METHODS
    # ============================================

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for summarization."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    def _format_scenes_for_chapter(self, scenes: list[dict]) -> str:
        """Format scene summaries for chapter summarization."""
        lines = []
        for i, scene in enumerate(scenes, 1):
            lines.append(f"SCENE {i}:")
            lines.append(f"Summary: {scene['summary']}")
            if scene.get("characters_present"):
                lines.append(f"Characters: {', '.join(scene['characters_present'])}")
            if scene.get("key_events"):
                lines.append(f"Key Events: {', '.join(scene['key_events'])}")
            lines.append(f"Mood: {scene.get('mood', 'unknown')}")
            lines.append("")
        return "\n".join(lines)

    def _format_chapters_for_arc(self, chapters: list[dict]) -> str:
        """Format chapter summaries for arc summarization."""
        lines = []
        for i, chapter in enumerate(chapters, 1):
            title = chapter.get("title", f"Chapter {i}")
            lines.append(f"CHAPTER {i}: {title}")
            lines.append(f"Summary: {chapter['summary']}")
            if chapter.get("major_events"):
                lines.append(f"Major Events: {', '.join(chapter['major_events'])}")
            if chapter.get("character_developments"):
                lines.append(f"Character Developments: {', '.join(chapter['character_developments'])}")
            if chapter.get("relationship_changes"):
                lines.append(f"Relationship Changes: {', '.join(chapter['relationship_changes'])}")
            lines.append("")
        return "\n".join(lines)

    def _get_session_id_by_number(self, story_id: UUID, session_number: int) -> str:
        """Get session UUID by session number."""
        result = self.db.table("sessions").select("id").eq(
            "story_id", str(story_id)
        ).eq("session_number", session_number).execute()

        if not result.data:
            raise ValueError(f"Session {session_number} not found for story {story_id}")
        return result.data[0]["id"]

    # ============================================
    # RETRIEVAL METHODS
    # ============================================

    def get_scene_summaries(
        self,
        story_id: UUID,
        session_id: Optional[UUID] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get scene summaries for a story or session."""
        query = self.db.table("scene_summaries").select("*").eq(
            "story_id", str(story_id)
        )
        if session_id:
            query = query.eq("session_id", str(session_id))

        return query.order("created_at", desc=True).limit(limit).execute().data

    def get_chapter_summaries(
        self,
        story_id: UUID,
        limit: int = 20,
    ) -> list[dict]:
        """Get chapter summaries for a story."""
        return self.db.table("chapter_summaries").select("*").eq(
            "story_id", str(story_id)
        ).order("chapter_number", desc=True).limit(limit).execute().data

    def get_arc_summaries(
        self,
        story_id: UUID,
        limit: int = 10,
    ) -> list[dict]:
        """Get arc summaries for a story."""
        return self.db.table("arc_summaries").select("*").eq(
            "story_id", str(story_id)
        ).order("arc_number", desc=True).limit(limit).execute().data

    def get_hierarchical_context(
        self,
        story_id: UUID,
        max_scenes: int = 5,
        max_chapters: int = 3,
        max_arcs: int = 2,
    ) -> dict:
        """
        Get hierarchical summaries for context building.

        Returns recent scenes, chapters, and arcs in a structured format.
        """
        scenes = self.get_scene_summaries(story_id, limit=max_scenes)
        chapters = self.get_chapter_summaries(story_id, limit=max_chapters)
        arcs = self.get_arc_summaries(story_id, limit=max_arcs)

        return {
            "recent_scenes": scenes,
            "recent_chapters": chapters,
            "story_arcs": arcs,
        }
