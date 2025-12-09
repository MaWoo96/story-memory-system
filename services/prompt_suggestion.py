"""
Grok-powered prompt suggestion service for image generation.

Uses xAI Grok to analyze character context and generate optimized
Stable Diffusion prompts for the Illustrious/Pony models.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from openai import OpenAI


class PromptSuggestion(BaseModel):
    """Structured output from Grok for image prompt generation."""

    positive_prompt: str = Field(
        description="Optimized positive prompt for Stable Diffusion. Include visual details, style tags, and quality markers."
    )
    negative_prompt: str = Field(
        description="Negative prompt to avoid unwanted elements."
    )
    recommended_style: str = Field(
        default="anime",
        description="Recommended style preset: anime, realistic, fantasy, or explicit"
    )
    recommended_pose: Optional[str] = Field(
        default=None,
        description="Suggested pose based on character and scene context"
    )
    recommended_expression: Optional[str] = Field(
        default=None,
        description="Suggested facial expression"
    )
    rationale: str = Field(
        description="Brief explanation of prompt choices"
    )


class ScenePromptSuggestion(BaseModel):
    """Structured output for scene image generation."""

    positive_prompt: str = Field(
        description="Optimized scene prompt including setting, lighting, and atmosphere"
    )
    negative_prompt: str = Field(
        description="Negative prompt for the scene"
    )
    participating_characters: list[str] = Field(
        default_factory=list,
        description="Characters that should appear in the scene"
    )
    recommended_style: str = Field(
        default="fantasy",
        description="Recommended style for the scene"
    )
    composition_notes: str = Field(
        description="Notes on scene composition and framing"
    )


class PromptSuggestionService:
    """
    Uses Grok to generate optimized image prompts based on story context.

    This service analyzes character descriptions, story themes, and recent
    narrative events to create prompts that are tailored to:
    - The Illustrious/Pony model ecosystem
    - The visual style of the story
    - Character-appropriate content and poses
    """

    SYSTEM_PROMPT = """You are an expert at generating Stable Diffusion prompts for anime-style character portraits.

You specialize in the Illustrious XL and Pony Diffusion V6 XL models, which work best with:

QUALITY TAGS (always include):
- masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest
- For Pony models: score_9, score_8_up, score_7_up

STYLE TAGS:
- source_anime for anime style
- Detailed eyes, detailed face, expressive features

BODY/CHARACTER TAGS:
- For mature female characters: milf, mature female, voluptuous body
- Include hair color, eye color, distinguishing features
- Clothing or state of dress based on context

NEGATIVE PROMPT ESSENTIALS:
- worst quality, low quality, bad anatomy, blurry
- censored, bar censor, mosaic (if NSFW)
- source_pony, source_furry (unless specifically wanted)

When generating prompts:
1. Extract visual details from the character description
2. Consider the story's tone and theme
3. Suggest appropriate poses and expressions
4. Keep prompts focused - too many tags reduce quality
5. Use natural language descriptions, not just tag lists

Output structured JSON with positive_prompt, negative_prompt, recommended_style, recommended_pose, recommended_expression, and rationale."""

    SCENE_SYSTEM_PROMPT = """You are an expert at generating Stable Diffusion prompts for scene illustrations.

Create prompts that capture:
- Setting and environment details
- Lighting and atmosphere
- Character positioning if characters are present
- Emotional tone of the scene

Use appropriate tags for the Illustrious/Pony model ecosystem.
Focus on composition and storytelling through visual elements."""

    def __init__(self):
        """Initialize the prompt suggestion service with xAI Grok."""
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        self.model = "grok-4-1-fast-reasoning"

    def suggest_character_prompt(
        self,
        character_name: str,
        character_description: str,
        character_facts: list[str],
        story_premise: Optional[str] = None,
        recent_events: Optional[str] = None,
        user_pose_hint: Optional[str] = None,
        user_expression_hint: Optional[str] = None,
        is_nsfw: bool = True,
    ) -> PromptSuggestion:
        """
        Generate an optimized image prompt for a character.

        Args:
            character_name: Name of the character
            character_description: Full description of the character
            character_facts: List of known facts about the character
            story_premise: The story's premise/setting
            recent_events: Recent narrative events for context
            user_pose_hint: User's optional pose preference
            user_expression_hint: User's optional expression preference
            is_nsfw: Whether to generate NSFW-appropriate prompts

        Returns:
            PromptSuggestion with optimized prompts and recommendations
        """
        # Build context message
        facts_str = "\n".join(f"- {fact}" for fact in character_facts) if character_facts else "No specific facts available."

        user_message = f"""Generate an optimized Stable Diffusion prompt for this character:

CHARACTER: {character_name}

DESCRIPTION:
{character_description}

KNOWN FACTS:
{facts_str}

"""
        if story_premise:
            user_message += f"""STORY CONTEXT:
{story_premise}

"""
        if recent_events:
            user_message += f"""RECENT EVENTS:
{recent_events}

"""
        if user_pose_hint:
            user_message += f"USER WANTS POSE: {user_pose_hint}\n"
        if user_expression_hint:
            user_message += f"USER WANTS EXPRESSION: {user_expression_hint}\n"

        user_message += f"""
CONTENT LEVEL: {"NSFW/Explicit content is allowed - include mature themes if appropriate for the character" if is_nsfw else "Keep content SFW/Safe"}

Generate a portrait prompt that captures this character's essence. Focus on visual details that would make a compelling portrait."""

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                response_format=PromptSuggestion,
                temperature=0.7,  # Some creativity for prompts
            )

            return completion.choices[0].message.parsed

        except Exception as e:
            print(f"[PromptSuggestion] Grok API error: {e}")
            # Return a reasonable default
            return PromptSuggestion(
                positive_prompt=f"masterpiece, best quality, {character_name}, {character_description[:200]}, detailed face, detailed eyes",
                negative_prompt="worst quality, low quality, bad anatomy, blurry",
                recommended_style="anime",
                recommended_pose=user_pose_hint,
                recommended_expression=user_expression_hint,
                rationale="Fallback prompt due to API error"
            )

    def suggest_scene_prompt(
        self,
        scene_description: str,
        participating_characters: list[dict],
        story_premise: Optional[str] = None,
        is_nsfw: bool = True,
    ) -> ScenePromptSuggestion:
        """
        Generate an optimized image prompt for a scene.

        Args:
            scene_description: Description of the scene
            participating_characters: List of characters in the scene with their info
            story_premise: The story's premise/setting
            is_nsfw: Whether to generate NSFW-appropriate prompts

        Returns:
            ScenePromptSuggestion with optimized prompts and recommendations
        """
        chars_str = ""
        for char in participating_characters:
            chars_str += f"- {char.get('name', 'Unknown')}: {char.get('description', 'No description')[:100]}\n"

        user_message = f"""Generate an optimized Stable Diffusion prompt for this scene:

SCENE DESCRIPTION:
{scene_description}

CHARACTERS IN SCENE:
{chars_str}
"""
        if story_premise:
            user_message += f"""
STORY CONTEXT:
{story_premise}
"""
        user_message += f"""
CONTENT LEVEL: {"NSFW/Explicit content is allowed" if is_nsfw else "Keep content SFW/Safe"}

Generate a scene prompt that captures this moment. Focus on composition, lighting, and atmosphere."""

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SCENE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                response_format=ScenePromptSuggestion,
                temperature=0.7,
            )

            return completion.choices[0].message.parsed

        except Exception as e:
            print(f"[PromptSuggestion] Grok API error: {e}")
            return ScenePromptSuggestion(
                positive_prompt=f"masterpiece, best quality, {scene_description[:200]}, detailed background, atmospheric",
                negative_prompt="worst quality, low quality, blurry",
                participating_characters=[c.get('name', 'Unknown') for c in participating_characters],
                recommended_style="fantasy",
                composition_notes="Fallback prompt due to API error"
            )

    def enhance_user_prompt(
        self,
        user_prompt: str,
        character_context: Optional[str] = None,
        style: str = "anime",
    ) -> str:
        """
        Enhance a user-provided prompt with quality tags and optimizations.

        Args:
            user_prompt: The user's raw prompt
            character_context: Optional character context to incorporate
            style: Target style

        Returns:
            Enhanced prompt string
        """
        # Quality prefix based on style
        quality_tags = {
            "anime": "masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest, source_anime",
            "realistic": "masterpiece, best quality, photorealistic, 8k uhd, detailed skin texture",
            "fantasy": "masterpiece, best quality, fantasy art, digital painting, dramatic lighting",
            "explicit": "masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest, detailed genitalia, anatomically correct",
        }

        prefix = quality_tags.get(style, quality_tags["anime"])

        # Combine
        enhanced = f"{prefix}, {user_prompt}"

        if character_context:
            enhanced = f"{enhanced}, {character_context}"

        return enhanced


# Singleton instance
_prompt_service: Optional[PromptSuggestionService] = None


def get_prompt_service() -> PromptSuggestionService:
    """Get or create the prompt suggestion service singleton."""
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptSuggestionService()
    return _prompt_service
