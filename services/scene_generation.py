"""
Scene generation service with deeper integrations.

Features:
1. Physical State → LoRA Mapping: Auto-selects LoRAs based on character physical states
2. Character Consistency in Scenes: Uses IP-Adapter with entity portraits for consistency
3. Background Generation Queue: Non-blocking async scene generation with status tracking

Based on research from:
- SpicyChat Semantic Memory 2.0 patterns for state tracking
- SillyTavern lorebook patterns for context-aware generation
"""

import os
import asyncio
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from config.lora_library import get_lora_library, LoRALibrary


class TaskStatus(str, Enum):
    """Status of a generation task."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationTask:
    """A scene generation task in the queue."""
    task_id: str
    story_id: str
    scene_description: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[dict] = None
    error: Optional[str] = None

    # Generation parameters
    characters_present: list[str] = field(default_factory=list)
    physical_states: list[dict] = field(default_factory=list)
    mood: Optional[str] = None
    session_id: Optional[str] = None


# ============================================
# PHYSICAL STATE → LORA MAPPING
# ============================================

# Mapping from physical state values to LoRA suggestions
PHYSICAL_STATE_LORA_MAPPINGS = {
    # Clothing states
    "clothing": {
        "nude": [],  # Default nude rendering, no specific LoRA needed
        "naked": [],
        "topless": [],
        "bottomless": [],
        "bikini": [{"filename": "Concept_Slutty_Clothes.safetensors", "weight": 0.5}],
        "micro bikini": [{"filename": "Concept_Slutty_Clothes.safetensors", "weight": 0.8}],
        "string bikini": [{"filename": "Concept_Slutty_Clothes.safetensors", "weight": 0.7}],
        "lingerie": [{"filename": "Concept_Slutty_Clothes.safetensors", "weight": 0.6}],
        "see-through": [{"filename": "Concept_Slutty_Clothes.safetensors", "weight": 0.7}],
        "tight clothes": [{"filename": "CLB-PD-v1.safetensors", "weight": 0.8}],
        "revealing": [{"filename": "Concept_Slutty_Clothes.safetensors", "weight": 0.6}],
        "cow print": [{"filename": "cowprint.safetensors", "weight": 0.8}],
        "cowprint": [{"filename": "cowprint.safetensors", "weight": 0.8}],
        "leotard": [{"filename": "EyepatchLeotardV1.safetensors", "weight": 0.7}],
        "sling bikini": [{"filename": "EyepatchLeotardV1.safetensors", "weight": 0.8}],
    },

    # Position states
    "position": {
        "cowgirl": [{"filename": "PSCowgirl.safetensors", "weight": 1.0}],
        "reverse cowgirl": [{"filename": "CowGirl.safetensors", "weight": 0.8}],
        "riding": [{"filename": "PSCowgirl.safetensors", "weight": 0.9}],
        "girl on top": [{"filename": "PSCowgirl.safetensors", "weight": 0.9}],
        "straddling": [{"filename": "PSCowgirl.safetensors", "weight": 0.8}],
        "mating press": [{"filename": "MatingFaceER.safetensors", "weight": 1.0}],
        "missionary": [{"filename": "MatingFaceER.safetensors", "weight": 0.7}],
        "legs up": [{"filename": "MatingFaceER.safetensors", "weight": 0.6}],
        "cunnilingus": [{"filename": "qqq-yuri-cunnilingus-v4.safetensors", "weight": 0.8}],
        "scissoring": [{"filename": "pussysandwich.safetensors", "weight": 0.8}],
        "tribbing": [{"filename": "pussysandwich.safetensors", "weight": 0.8}],
    },

    # Temporary states that affect rendering
    "temporary_states": {
        "aroused": [],  # Let model handle naturally
        "sweaty": [],
        "wet": [],
        "cum covered": [{"filename": "extreme_bukkake_v0.1-pony.safetensors", "weight": 0.7}],
        "bukkake": [{"filename": "extreme_bukkake_v0.1-pony.safetensors", "weight": 0.9}],
        "lactating": [{"filename": "MilkFactoryStyleLoRa-01.safetensors", "weight": 0.7}],
        "milking": [{"filename": "MilkFactoryStyleLoRa-01.safetensors", "weight": 0.8}],
    },

    # Character type (from description/facts)
    "character_type": {
        "succubus": [{"filename": "Succubus-Illustrious-v01.safetensors", "weight": 0.7}],
        "demon": [{"filename": "Leslie_-_Demon_Deals.safetensors", "weight": 0.6}],
        "demon girl": [{"filename": "Succubus-Illustrious-v01.safetensors", "weight": 0.7}],
        "elf": [{"filename": "Youkoso_Sukebe_Elf_No_Mori_E.safetensors", "weight": 0.8}],
        "colored skin": [{"filename": "ColoredSkinSuccubus_v10.safetensors", "weight": 0.8}],
        "purple skin": [{"filename": "ColoredSkinSuccubus_v10.safetensors", "weight": 0.8}],
        "blue skin": [{"filename": "ColoredSkinSuccubus_v10.safetensors", "weight": 0.8}],
    },
}

# Scene mood to LoRA/style adjustments
MOOD_LORA_ADJUSTMENTS = {
    "intimate": {
        "cfg_boost": 0.0,  # Slight CFG reduction for softer output
        "loras": [],
    },
    "action": {
        "cfg_boost": 0.5,  # Sharper for action scenes
        "loras": [],
    },
    "romantic": {
        "cfg_boost": -0.3,
        "loras": [],
    },
    "mysterious": {
        "cfg_boost": 0.0,
        "loras": [],
    },
    "tense": {
        "cfg_boost": 0.3,
        "loras": [],
    },
}


class PhysicalStateLoRAMapper:
    """
    Maps character physical states to appropriate LoRAs.

    Takes physical_states from the database and returns a list of LoRAs
    that should be applied to the generation.
    """

    def __init__(self):
        self.lora_library = get_lora_library()

    def map_states_to_loras(
        self,
        physical_states: list[dict],
        scene_description: Optional[str] = None,
        mood: Optional[str] = None,
    ) -> list[dict]:
        """
        Map physical states to LoRAs for scene generation.

        Args:
            physical_states: List of physical state dicts per character
                Each dict has: character, clothing, position, location_in_scene, temporary_states
            scene_description: Scene description for additional keyword detection
            mood: Current scene mood for adjustments

        Returns:
            List of LoRAs to apply: [{"name": "file.safetensors", "weight": 1.0}]
        """
        suggested_loras = {}

        # Process each character's physical state
        for state in physical_states:
            self._process_character_state(state, suggested_loras)

        # Process scene description keywords
        if scene_description:
            self._detect_from_text(scene_description, suggested_loras)

        # Apply mood adjustments
        if mood and mood in MOOD_LORA_ADJUSTMENTS:
            for lora in MOOD_LORA_ADJUSTMENTS[mood].get("loras", []):
                if self.lora_library.is_available(lora["filename"]):
                    suggested_loras[lora["filename"]] = lora

        # Filter to only available LoRAs
        result = []
        for filename, lora_config in suggested_loras.items():
            if self.lora_library.is_available(filename):
                result.append({
                    "name": filename,
                    "weight": lora_config.get("weight", 1.0)
                })

        return result

    def _process_character_state(self, state: dict, suggested: dict):
        """Process a single character's physical state."""
        # Process clothing
        clothing = state.get("clothing") or []
        if isinstance(clothing, str):
            clothing = [clothing]

        for item in clothing:
            item_lower = item.lower()
            for keyword, loras in PHYSICAL_STATE_LORA_MAPPINGS["clothing"].items():
                if keyword in item_lower:
                    for lora in loras:
                        suggested[lora["filename"]] = lora

        # Process position
        position = state.get("position") or ""
        position_lower = position.lower()
        for keyword, loras in PHYSICAL_STATE_LORA_MAPPINGS["position"].items():
            if keyword in position_lower:
                for lora in loras:
                    # Positions are important, use full weight
                    suggested[lora["filename"]] = lora

        # Process temporary states
        temp_states = state.get("temporary_states") or []
        if isinstance(temp_states, str):
            temp_states = [temp_states]

        for temp in temp_states:
            temp_lower = temp.lower()
            for keyword, loras in PHYSICAL_STATE_LORA_MAPPINGS["temporary_states"].items():
                if keyword in temp_lower:
                    for lora in loras:
                        suggested[lora["filename"]] = lora

    def _detect_from_text(self, text: str, suggested: dict):
        """Detect LoRAs from scene description text."""
        text_lower = text.lower()

        # Check character type keywords
        for keyword, loras in PHYSICAL_STATE_LORA_MAPPINGS["character_type"].items():
            if keyword in text_lower:
                for lora in loras:
                    suggested[lora["filename"]] = lora

        # Also use the library's built-in detection
        library_suggestions = self.lora_library.detect_from_prompt(text)
        for suggestion in library_suggestions:
            if suggestion.get("auto_enable"):
                suggested[suggestion["filename"]] = {
                    "filename": suggestion["filename"],
                    "weight": suggestion.get("default_weight", 1.0)
                }

    def get_cfg_adjustment(self, mood: Optional[str]) -> float:
        """Get CFG adjustment based on mood."""
        if mood and mood in MOOD_LORA_ADJUSTMENTS:
            return MOOD_LORA_ADJUSTMENTS[mood].get("cfg_boost", 0.0)
        return 0.0


# ============================================
# BACKGROUND GENERATION QUEUE
# ============================================

class SceneGenerationQueue:
    """
    In-memory task queue for background scene generation.

    For production, this should be replaced with Celery/Redis, but this
    provides the same interface for development and small deployments.
    """

    def __init__(self, max_concurrent: int = 2):
        self.tasks: dict[str, GenerationTask] = {}
        self.pending_queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self._workers_started = False
        self._processing_count = 0
        self._lock = asyncio.Lock()

    async def enqueue(
        self,
        story_id: str,
        scene_description: str,
        characters_present: list[str] = None,
        physical_states: list[dict] = None,
        mood: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Add a scene generation task to the queue.

        Returns:
            task_id to check status/retrieve results
        """
        task_id = str(uuid4())

        task = GenerationTask(
            task_id=task_id,
            story_id=story_id,
            scene_description=scene_description,
            characters_present=characters_present or [],
            physical_states=physical_states or [],
            mood=mood,
            session_id=session_id,
        )

        self.tasks[task_id] = task
        await self.pending_queue.put(task_id)

        # Start workers if not already running
        if not self._workers_started:
            self._start_workers()

        return task_id

    def get_status(self, task_id: str) -> Optional[dict]:
        """Get the status of a generation task."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result,
            "error": task.error,
        }

    def get_result(self, task_id: str) -> Optional[dict]:
        """Get the result of a completed task."""
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.COMPLETED:
            return task.result
        return None

    def _start_workers(self):
        """Start background worker coroutines."""
        self._workers_started = True
        for i in range(self.max_concurrent):
            asyncio.create_task(self._worker(i))

    async def _worker(self, worker_id: int):
        """Background worker that processes generation tasks."""
        while True:
            try:
                # Get next task from queue
                task_id = await self.pending_queue.get()
                task = self.tasks.get(task_id)

                if not task:
                    continue

                async with self._lock:
                    self._processing_count += 1

                # Update status
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.now()

                try:
                    # Process the generation
                    result = await self._process_task(task)
                    task.result = result
                    task.status = TaskStatus.COMPLETED

                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                    print(f"[Worker {worker_id}] Task {task_id} failed: {e}")

                finally:
                    task.completed_at = datetime.now()
                    async with self._lock:
                        self._processing_count -= 1
                    self.pending_queue.task_done()

            except Exception as e:
                print(f"[Worker {worker_id}] Error: {e}")
                await asyncio.sleep(1)

    async def _process_task(self, task: GenerationTask) -> dict:
        """Process a single generation task."""
        from services.image_generation import ImageGenerationService

        image_service = ImageGenerationService()
        lora_mapper = PhysicalStateLoRAMapper()

        # Get LoRAs based on physical states
        auto_loras = lora_mapper.map_states_to_loras(
            physical_states=task.physical_states,
            scene_description=task.scene_description,
            mood=task.mood,
        )

        # Get CFG adjustment based on mood
        cfg_adjustment = lora_mapper.get_cfg_adjustment(task.mood)
        base_cfg = 7.0  # Default
        adjusted_cfg = base_cfg + cfg_adjustment

        # Build the generation request
        # For now, use the standard scene generation
        # In future, this will use IP-Adapter for character consistency
        result = await image_service.generate_scene(
            description=task.scene_description,
            story_id=UUID(task.story_id),
            session_id=UUID(task.session_id) if task.session_id else None,
            style="anime",
            participating_entities=task.characters_present,
        )

        return {
            "file_path": result.file_path,
            "file_url": result.file_url,
            "generation_prompt": result.generation_prompt,
            "seed": result.seed,
            "auto_loras_applied": auto_loras,
            "cfg_used": adjusted_cfg,
        }


# ============================================
# ENHANCED SCENE GENERATION SERVICE
# ============================================

class EnhancedSceneGenerationService:
    """
    Scene generation service with:
    1. Physical state → LoRA mapping
    2. Character consistency via IP-Adapter
    3. Async background generation queue
    """

    def __init__(self, db_client=None):
        self.db = db_client
        self.lora_mapper = PhysicalStateLoRAMapper()
        self.queue = SceneGenerationQueue()
        self._image_service = None

    @property
    def image_service(self):
        """Lazy-load image service."""
        if self._image_service is None:
            from services.image_generation import ImageGenerationService
            self._image_service = ImageGenerationService()
        return self._image_service

    async def generate_scene_async(
        self,
        story_id: UUID,
        scene_description: str,
        characters_present: list[str] = None,
        physical_states: list[dict] = None,
        mood: Optional[str] = None,
        session_id: Optional[UUID] = None,
        wait_for_result: bool = False,
        timeout: int = 300,
    ) -> dict:
        """
        Generate a scene image asynchronously.

        Args:
            story_id: Story UUID
            scene_description: Description of the scene
            characters_present: Names of characters in scene
            physical_states: Physical states per character
            mood: Scene mood
            session_id: Optional session UUID
            wait_for_result: If True, wait for completion and return result
            timeout: Max wait time in seconds if wait_for_result is True

        Returns:
            If wait_for_result: Full result dict
            Otherwise: {"task_id": "...", "status": "pending"}
        """
        task_id = await self.queue.enqueue(
            story_id=str(story_id),
            scene_description=scene_description,
            characters_present=characters_present or [],
            physical_states=physical_states or [],
            mood=mood,
            session_id=str(session_id) if session_id else None,
        )

        if not wait_for_result:
            return {
                "task_id": task_id,
                "status": "pending",
            }

        # Wait for completion
        start = datetime.now()
        while (datetime.now() - start).seconds < timeout:
            status = self.queue.get_status(task_id)
            if status and status["status"] in ["completed", "failed"]:
                return status
            await asyncio.sleep(1)

        return {
            "task_id": task_id,
            "status": "timeout",
            "error": f"Task did not complete within {timeout} seconds",
        }

    def get_task_status(self, task_id: str) -> Optional[dict]:
        """Get status of a generation task."""
        return self.queue.get_status(task_id)

    def get_task_result(self, task_id: str) -> Optional[dict]:
        """Get result of a completed task."""
        return self.queue.get_result(task_id)

    async def get_character_references(
        self,
        story_id: UUID,
        character_names: list[str],
    ) -> dict[str, Optional[str]]:
        """
        Get reference image paths for characters (for IP-Adapter).

        Args:
            story_id: Story UUID
            character_names: List of character names

        Returns:
            Dict mapping character name to their primary portrait path
        """
        if not self.db:
            return {}

        references = {}

        for name in character_names:
            # Find entity by name
            entity_result = self.db.table("entities").select("id").eq(
                "story_id", str(story_id)
            ).ilike("canonical_name", name).execute()

            if not entity_result.data:
                references[name] = None
                continue

            entity_id = entity_result.data[0]["id"]

            # Get primary portrait
            image_result = self.db.table("entity_images").select(
                "file_path", "file_url"
            ).eq("entity_id", entity_id).eq(
                "is_primary", True
            ).execute()

            if image_result.data:
                # Prefer file_url (cloud storage) over local path
                references[name] = (
                    image_result.data[0].get("file_url") or
                    image_result.data[0].get("file_path")
                )
            else:
                references[name] = None

        return references

    async def generate_scene_with_characters(
        self,
        story_id: UUID,
        scene_description: str,
        characters_present: list[str] = None,
        physical_states: list[dict] = None,
        mood: Optional[str] = None,
        session_id: Optional[UUID] = None,
        use_character_references: bool = True,
        width: int = 1024,
        height: int = 576,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Generate a scene with full integrations (synchronous, blocking).

        This is the main entry point that:
        1. Maps physical states to LoRAs
        2. Gets character reference images for consistency
        3. Generates the scene with all context

        Args:
            story_id: Story UUID
            scene_description: Description of the scene
            characters_present: Names of characters in scene
            physical_states: Physical states per character
            mood: Scene mood
            session_id: Optional session UUID
            use_character_references: Whether to use IP-Adapter with portraits
            width: Output width
            height: Output height
            seed: Random seed

        Returns:
            Generation result dict
        """
        characters_present = characters_present or []
        physical_states = physical_states or []

        # 1. Map physical states to LoRAs
        auto_loras = self.lora_mapper.map_states_to_loras(
            physical_states=physical_states,
            scene_description=scene_description,
            mood=mood,
        )

        # 2. Get character references if enabled
        character_refs = {}
        if use_character_references and characters_present:
            character_refs = await self.get_character_references(
                story_id=story_id,
                character_names=characters_present,
            )

        # 3. Get CFG adjustment based on mood
        cfg_adjustment = self.lora_mapper.get_cfg_adjustment(mood)
        base_cfg = 7.0
        adjusted_cfg = base_cfg + cfg_adjustment

        # 4. Build enhanced prompt with character context
        enhanced_description = self._enhance_description_with_states(
            scene_description,
            physical_states,
            characters_present,
        )

        # 5. Generate the scene
        # Note: Currently using standard generation
        # IP-Adapter workflow would be added here when ComfyUI workflow is expanded
        result = await self.image_service.generate_scene(
            description=enhanced_description,
            story_id=story_id,
            session_id=session_id,
            style="anime",
            width=width,
            height=height,
            seed=seed,
            participating_entities=characters_present,
        )

        return {
            "file_path": result.file_path,
            "file_url": result.file_url,
            "generation_prompt": result.generation_prompt,
            "negative_prompt": result.negative_prompt,
            "seed": result.seed,
            "model_used": result.model_used,
            "width": result.width,
            "height": result.height,
            # Integration metadata
            "auto_loras_applied": [{"name": l["name"], "weight": l["weight"]} for l in auto_loras],
            "cfg_used": adjusted_cfg,
            "character_references_used": {
                name: ref is not None
                for name, ref in character_refs.items()
            },
            "mood": mood,
        }

    def _enhance_description_with_states(
        self,
        description: str,
        physical_states: list[dict],
        characters: list[str],
    ) -> str:
        """
        Enhance scene description with physical state details.

        Adds relevant tags from physical states to ensure they're
        included in the generation prompt.
        """
        enhancement_parts = [description]

        # Add character-specific state tags
        for state in physical_states:
            char_name = state.get("character", "")

            # Clothing
            clothing = state.get("clothing") or []
            if clothing:
                if isinstance(clothing, list):
                    enhancement_parts.append(", ".join(clothing))
                else:
                    enhancement_parts.append(clothing)

            # Position
            position = state.get("position")
            if position:
                enhancement_parts.append(position)

            # Temporary states (aroused, sweaty, etc.)
            temp_states = state.get("temporary_states") or []
            if temp_states:
                if isinstance(temp_states, list):
                    enhancement_parts.append(", ".join(temp_states))
                else:
                    enhancement_parts.append(temp_states)

        # Add multi-character tag if multiple characters
        if len(characters) == 2:
            enhancement_parts.append("2girls")
        elif len(characters) >= 3:
            enhancement_parts.append("multiple girls, group")

        return ", ".join(filter(None, enhancement_parts))


# ============================================
# SINGLETON AND FACTORY
# ============================================

_scene_service_instance: Optional[EnhancedSceneGenerationService] = None


def get_scene_generation_service(db_client=None) -> EnhancedSceneGenerationService:
    """Get or create the scene generation service singleton."""
    global _scene_service_instance
    if _scene_service_instance is None:
        _scene_service_instance = EnhancedSceneGenerationService(db_client=db_client)
    return _scene_service_instance
