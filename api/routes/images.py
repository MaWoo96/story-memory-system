"""
Image generation API endpoints.

Handles character portrait generation, scene illustrations, and image management.
"""

import os
import httpx
import json
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from uuid import UUID
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

from api.dependencies import get_supabase_client
from schemas.extraction import Entity, EntityType, Fact, FactType


router = APIRouter(prefix="/api/images", tags=["images"])


# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class GeneratePortraitRequest(BaseModel):
    """Request to generate a character portrait."""
    style: str = Field(default="anime", description="Art style: anime, realistic, fantasy")
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=768, ge=256, le=1024)
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    additional_tags: Optional[list[str]] = Field(default=None, description="Extra prompt tags")
    pose: Optional[str] = Field(default=None, description="Specific pose")
    expression: Optional[str] = Field(default=None, description="Facial expression")
    use_reference: bool = Field(default=True, description="Use existing portrait as reference")


class GenerateSceneRequest(BaseModel):
    """Request to generate a scene illustration."""
    description: str = Field(description="Scene description")
    style: str = Field(default="fantasy")
    width: int = Field(default=1024, ge=256, le=1920)
    height: int = Field(default=576, ge=256, le=1080)
    seed: Optional[int] = Field(default=None)
    session_id: Optional[UUID] = Field(default=None)
    event_id: Optional[UUID] = Field(default=None)
    participating_entities: Optional[list[str]] = Field(default=None)


class ImageResponse(BaseModel):
    """Response with generated image info."""
    id: Optional[UUID] = None
    file_path: str
    file_url: Optional[str] = None
    generation_prompt: str
    seed: int
    model_used: str
    width: int
    height: int


class ComfyUIStatusResponse(BaseModel):
    """ComfyUI server status."""
    status: str
    message: Optional[str] = None
    available_models: Optional[list[str]] = None


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_image_service():
    """Get image generation service instance."""
    from services.image_generation import ImageGenerationService
    return ImageGenerationService()


def get_consistency_service():
    """Get character consistency service instance."""
    from services.image_generation import ImageGenerationService, CharacterConsistencyService
    db = get_supabase_client()
    image_service = ImageGenerationService()
    return CharacterConsistencyService(image_service=image_service, db_client=db)


async def entity_to_extraction_entity(entity_data: dict, db) -> Entity:
    """Convert database entity to extraction Entity model."""
    # Get facts for this entity
    facts_result = db.table("entity_facts").select(
        "fact_type, fact_value, importance"
    ).eq("entity_id", entity_data["id"]).is_(
        "invalidated_session_id", "null"
    ).execute()

    facts = [
        Fact(
            fact_type=FactType(f["fact_type"]),
            fact_value=f["fact_value"],
            importance=f["importance"]
        )
        for f in facts_result.data
    ]

    # Get aliases
    aliases_result = db.table("entity_aliases").select("alias").eq(
        "entity_id", entity_data["id"]
    ).execute()
    aliases = [a["alias"] for a in aliases_result.data]

    return Entity(
        entity_type=EntityType(entity_data["entity_type"]),
        canonical_name=entity_data["canonical_name"],
        aliases=aliases,
        description=entity_data.get("description", ""),
        facts=facts,
        importance=entity_data.get("computed_importance", 0.5),
    )


# ============================================
# ENDPOINTS
# ============================================

@router.get("/status", response_model=ComfyUIStatusResponse)
async def get_comfyui_status():
    """Check ComfyUI server status and available models."""
    image_service = get_image_service()

    status = await image_service.check_comfyui_status()

    if status["status"] == "connected":
        models = await image_service.list_available_models()
        return ComfyUIStatusResponse(
            status="connected",
            available_models=models
        )

    return ComfyUIStatusResponse(
        status=status["status"],
        message=status.get("message")
    )


@router.post("/entities/{entity_id}/portrait", response_model=ImageResponse)
async def generate_entity_portrait(
    entity_id: UUID,
    request: GeneratePortraitRequest,
    background_tasks: BackgroundTasks,
):
    """
    Generate a portrait for an entity using local Stable Diffusion.
    Uses entity description and facts to build the prompt.
    """
    db = get_supabase_client()

    # Get entity
    entity_result = db.table("entities").select("*").eq(
        "id", str(entity_id)
    ).execute()

    if not entity_result.data:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

    entity_data = entity_result.data[0]
    story_id = UUID(entity_data["story_id"])

    # Convert to Entity model
    entity = await entity_to_extraction_entity(entity_data, db)

    # Generate portrait
    try:
        if request.use_reference:
            consistency_service = get_consistency_service()
            result = await consistency_service.generate_consistent_portrait(
                entity=entity,
                entity_id=entity_id,
                story_id=story_id,
                style=request.style,
                width=request.width,
                height=request.height,
                seed=request.seed,
                additional_tags=request.additional_tags,
                pose=request.pose,
                expression=request.expression,
            )
        else:
            image_service = get_image_service()
            result = await image_service.generate_portrait(
                entity=entity,
                entity_id=entity_id,
                story_id=story_id,
                style=request.style,
                width=request.width,
                height=request.height,
                seed=request.seed,
                additional_tags=request.additional_tags,
                pose=request.pose,
                expression=request.expression,
            )

            # Store in database
            db.table("entity_images").insert({
                "entity_id": str(entity_id),
                "image_type": "portrait",
                "file_path": result.file_path,
                "file_url": result.file_url,
                "generation_prompt": result.generation_prompt,
                "negative_prompt": result.negative_prompt,
                "seed": result.seed,
                "model_used": result.model_used,
                "sampler": result.sampler,
                "steps": result.steps,
                "cfg_scale": result.cfg_scale,
                "width": result.width,
                "height": result.height,
                "is_primary": False,
            }).execute()

        return ImageResponse(
            file_path=result.file_path,
            file_url=result.file_url,
            generation_prompt=result.generation_prompt,
            seed=result.seed,
            model_used=result.model_used,
            width=result.width,
            height=result.height,
        )

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/stories/{story_id}/scene", response_model=ImageResponse)
async def generate_scene_image(
    story_id: UUID,
    request: GenerateSceneRequest,
):
    """Generate a scene illustration for a story."""
    db = get_supabase_client()

    # Verify story exists
    story_result = db.table("stories").select("id").eq(
        "id", str(story_id)
    ).execute()

    if not story_result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    try:
        image_service = get_image_service()
        result = await image_service.generate_scene(
            description=request.description,
            story_id=story_id,
            session_id=request.session_id,
            event_id=request.event_id,
            style=request.style,
            width=request.width,
            height=request.height,
            seed=request.seed,
            participating_entities=request.participating_entities,
        )

        # Store in database
        insert_data = {
            "story_id": str(story_id),
            "image_type": "scene",
            "file_path": result.file_path,
            "file_url": result.file_url,
            "generation_prompt": result.generation_prompt,
            "negative_prompt": result.negative_prompt,
            "seed": result.seed,
            "model_used": result.model_used,
        }

        if request.session_id:
            insert_data["session_id"] = str(request.session_id)
        if request.event_id:
            insert_data["event_id"] = str(request.event_id)
        if request.participating_entities:
            # Resolve entity IDs
            entity_ids = []
            for name in request.participating_entities:
                entity = db.table("entities").select("id").eq(
                    "story_id", str(story_id)
                ).ilike("canonical_name", name).execute()
                if entity.data:
                    entity_ids.append(entity.data[0]["id"])
            if entity_ids:
                insert_data["participating_entities"] = entity_ids

        db.table("scene_images").insert(insert_data).execute()

        return ImageResponse(
            file_path=result.file_path,
            file_url=result.file_url,
            generation_prompt=result.generation_prompt,
            seed=result.seed,
            model_used=result.model_used,
            width=result.width,
            height=result.height,
        )

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.get("/entities/{entity_id}/images")
async def get_entity_images(
    entity_id: UUID,
    image_type: Optional[str] = Query(default=None, description="Filter by type: portrait, full_body, expression"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Get all generated images for an entity."""
    db = get_supabase_client()

    query = db.table("entity_images").select("*").eq(
        "entity_id", str(entity_id)
    )

    if image_type:
        query = query.eq("image_type", image_type)

    result = query.order("created_at", desc=True).limit(limit).execute()

    return {
        "entity_id": str(entity_id),
        "count": len(result.data),
        "images": result.data
    }


@router.get("/stories/{story_id}/images")
async def get_story_images(
    story_id: UUID,
    image_type: Optional[str] = Query(default=None, description="Filter by type: scene, action, location"),
    session_id: Optional[UUID] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Get all scene images for a story."""
    db = get_supabase_client()

    query = db.table("scene_images").select("*").eq(
        "story_id", str(story_id)
    )

    if image_type:
        query = query.eq("image_type", image_type)
    if session_id:
        query = query.eq("session_id", str(session_id))

    result = query.order("created_at", desc=True).limit(limit).execute()

    return {
        "story_id": str(story_id),
        "count": len(result.data),
        "images": result.data
    }


@router.post("/entities/{entity_id}/images/{image_id}/set-primary")
async def set_primary_image(
    entity_id: UUID,
    image_id: UUID,
):
    """Set an image as the primary portrait for an entity."""
    db = get_supabase_client()

    # Verify image exists and belongs to entity
    image_result = db.table("entity_images").select("id").eq(
        "id", str(image_id)
    ).eq("entity_id", str(entity_id)).execute()

    if not image_result.data:
        raise HTTPException(status_code=404, detail="Image not found for this entity")

    # Unset current primary
    db.table("entity_images").update({
        "is_primary": False
    }).eq("entity_id", str(entity_id)).eq("is_primary", True).execute()

    # Set new primary
    db.table("entity_images").update({
        "is_primary": True
    }).eq("id", str(image_id)).execute()

    return {"success": True, "primary_image_id": str(image_id)}


@router.delete("/images/{image_id}")
async def delete_image(image_id: UUID):
    """Delete an image (soft delete - keeps file, removes from DB)."""
    db = get_supabase_client()

    # Check entity_images first
    entity_image = db.table("entity_images").select("id, file_path").eq(
        "id", str(image_id)
    ).execute()

    if entity_image.data:
        db.table("entity_images").delete().eq("id", str(image_id)).execute()
        return {"success": True, "deleted_from": "entity_images"}

    # Check scene_images
    scene_image = db.table("scene_images").select("id, file_path").eq(
        "id", str(image_id)
    ).execute()

    if scene_image.data:
        db.table("scene_images").delete().eq("id", str(image_id)).execute()
        return {"success": True, "deleted_from": "scene_images"}

    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/styles")
async def get_available_styles():
    """Get available image generation styles and their descriptions."""
    from services.image_generation import ImageGenerationService

    styles = {}
    for name, preset in ImageGenerationService.STYLE_PRESETS.items():
        styles[name] = {
            "name": name.title(),
            "model": preset["model"],
            "sampler": preset["sampler"],
            "steps": preset["steps"],
            "cfg_scale": preset["cfg"],
            "quality_tags": preset["quality_tags"][:5],  # Preview
        }

    return {"styles": styles}


# ============================================
# DIRECT TEST ENDPOINT (no database required)
# ============================================

class DirectGenerateRequest(BaseModel):
    """Direct image generation request - bypasses database."""
    canonical_name: str = Field(description="Character name")
    description: str = Field(description="Full visual description")
    facts: list[str] = Field(default=[], description="Character facts")
    style: str = Field(default="fantasy", description="Art style: anime, realistic, fantasy")
    story_id: str = Field(default="test", description="Story ID for style matching")
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=768, ge=256, le=1024)
    seed: Optional[int] = Field(default=None)


@router.post("/generate-direct")
async def generate_image_direct(request: DirectGenerateRequest):
    """
    Generate an image directly without database.
    Perfect for testing the image generation pipeline.

    Story ID keywords trigger different styles:
    - "isekai", "mira", "eros", "mana", "test", "aurelia" → isekai_milf style
    - "grimdark", "warrior", "battle" → grimdark_warrior style
    - "romance", "wholesome" → wholesome_romance style
    - "dark", "erotic", "sensual" → dark_fantasy_erotic style
    """
    from services.image_generation import ImageGenerationService
    from schemas.extraction import Entity, EntityType, Fact, FactType
    from uuid import uuid4

    # Build Entity from request
    facts = [
        Fact(fact_type=FactType.TRAIT, fact_value=f, importance=0.8)
        for f in request.facts
    ]

    entity = Entity(
        entity_type=EntityType.CHARACTER,
        canonical_name=request.canonical_name,
        aliases=[],
        description=request.description,
        facts=facts,
        importance=0.9,
    )

    # Generate image
    try:
        image_service = ImageGenerationService()

        # Generate fake UUIDs for file organization
        story_uuid = uuid4()
        entity_uuid = uuid4()

        result = await image_service.generate_portrait(
            entity=entity,
            entity_id=entity_uuid,
            story_id=story_uuid,
            style=request.style,
            width=request.width,
            height=request.height,
            seed=request.seed,
        )

        # Override story_id in prompt building by regenerating with correct story context
        positive, negative = image_service.build_character_prompt(
            entity=entity,
            style=request.style,
            story_id=request.story_id,
        )

        return {
            "success": True,
            "file_path": result.file_path,
            "file_url": result.file_url,
            "prompt_used": positive,
            "negative_prompt": negative,
            "seed": result.seed,
            "model": result.model_used,
            "story_style_detected": image_service.get_story_style(request.story_id),
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ============================================
# RAW PROMPT ENDPOINT (exact prompt, no modifications)
# ============================================

class RawGenerateRequest(BaseModel):
    """Raw image generation request - uses exact prompt with no modifications."""
    positive_prompt: str = Field(description="Exact positive prompt to use")
    negative_prompt: str = Field(
        default="worst aesthetic, worst quality, low quality, bad quality, lowres, bar censor, censored",
        description="Exact negative prompt to use"
    )
    width: int = Field(default=640, ge=256, le=1920)
    height: int = Field(default=1024, ge=256, le=1920)
    seed: Optional[int] = Field(default=None)
    steps: int = Field(default=30, ge=1, le=150)
    cfg: float = Field(default=5.0, ge=1.0, le=30.0)
    sampler: str = Field(default="euler_ancestral", description="Sampler name")
    scheduler: str = Field(default="sgm_uniform", description="Scheduler name")


@router.post("/generate-raw")
async def generate_image_raw(request: RawGenerateRequest):
    """
    Generate an image with EXACT prompt - no story styles, no quality tags added.
    Perfect for testing specific prompts from CivitAI or other sources.

    Uses: CFG 5, Euler a, SGM Uniform, CLIP Skip 2, Zheng LoRA at 1.0
    """
    from services.image_generation import ImageGenerationService
    from uuid import uuid4
    import random

    try:
        image_service = ImageGenerationService()

        # Use provided seed or generate random
        seed = request.seed or random.randint(0, 2**32 - 1)

        # Build workflow with exact prompt
        workflow = image_service._build_txt2img_workflow(
            positive=request.positive_prompt,
            negative=request.negative_prompt,
            width=request.width,
            height=request.height,
            seed=seed,
            model=image_service.DEFAULT_MODEL,
            sampler=request.sampler,
            steps=request.steps,
            cfg=request.cfg,
            scheduler=request.scheduler,
            use_loras=True,  # Use Zheng + other LoRAs
        )

        # Execute generation
        image_data = await image_service._execute_workflow(workflow)

        # Save to filesystem
        story_uuid = uuid4()
        entity_uuid = uuid4()
        output_dir = image_service.output_base / str(story_uuid) / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_{seed}_{timestamp}.png"
        file_path = output_dir / filename

        with open(file_path, "wb") as f:
            f.write(image_data)

        # Upload to B2 if available
        file_url = None
        if image_service._b2_storage:
            try:
                b2_result = image_service._b2_storage.upload_image(
                    file_data=image_data,
                    story_id=str(story_uuid),
                    entity_id=str(entity_uuid),
                    image_type="raw",
                    style="raw",
                    extension="png"
                )
                file_url = b2_result["file_url"]
            except Exception as e:
                print(f"[ImageGen] B2 upload failed: {e}")

        return {
            "success": True,
            "file_path": str(file_path),
            "file_url": file_url,
            "prompt_used": request.positive_prompt,
            "negative_prompt": request.negative_prompt,
            "seed": seed,
            "model": image_service.DEFAULT_MODEL,
            "settings": {
                "steps": request.steps,
                "cfg": request.cfg,
                "sampler": request.sampler,
                "scheduler": request.scheduler,
                "width": request.width,
                "height": request.height,
            }
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ============================================
# RAW PROMPT WITH HIRES UPSCALING
# ============================================

class RawHiResRequest(BaseModel):
    """Raw image generation with HiRes upscaling - Illustrious Animilf standard.

    OPTIMAL SETTINGS (tested 2024-12-08):
    - sampler: euler_ancestral (better for expressive faces)
    - scheduler: normal
    - steps: 28
    - cfg: 7.0
    - hires_steps: 12
    - hires_denoise: 0.3

    These settings produce sharp, detailed images with expressive faces.
    """
    positive_prompt: str = Field(description="Exact positive prompt to use")
    negative_prompt: str = Field(
        default="worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, signature, watermarks, username, blurry, artist name, loli, child, flat chest, petite, closed mouth",
        description="Exact negative prompt to use"
    )
    width: int = Field(default=832, ge=256, le=1920, description="Base width (will be upscaled)")
    height: int = Field(default=1216, ge=256, le=1920, description="Base height (will be upscaled)")
    seed: Optional[int] = Field(default=None)
    steps: int = Field(default=28, ge=1, le=150, description="Base generation steps")
    cfg: float = Field(default=7.0, ge=1.0, le=30.0, description="CFG scale (7.0 for Illustrious)")
    sampler: str = Field(default="euler_ancestral", description="Sampler name (euler_ancestral = best for faces)")
    scheduler: str = Field(default="normal", description="Scheduler name")
    # HiRes settings - optimized for crisp output
    hires_scale: float = Field(default=1.5, ge=1.0, le=4.0, description="Upscale factor (1.5 = 50% larger)")
    hires_steps: int = Field(default=12, ge=1, le=50, description="HiRes refinement steps")
    hires_denoise: float = Field(default=0.3, ge=0.0, le=1.0, description="HiRes denoise strength (0.3 = sharp)")
    upscaler: str = Field(default="4x-UltraSharp.pth", description="Upscaler model name (unused with latent upscale)")
    # Custom model/LoRA settings
    model: Optional[str] = Field(default=None, description="Checkpoint model (default: illustriousAnimilf_v01)")
    custom_loras: Optional[list[dict]] = Field(default=None, description="Custom LoRAs: [{'name': 'lora.safetensors', 'weight': 1.0}]")
    use_default_loras: bool = Field(default=False, description="Use Pony LoRA chain (legacy)")
    use_standard_lora: bool = Field(default=True, description="Use Youkoso Elf + Breast Slider LoRA chain (standard)")
    # Quality enhancements
    use_fp16_vae: bool = Field(default=True, description="Use FP16-fix VAE for better Illustrious/SDXL quality")
    use_face_detailer: bool = Field(default=False, description="Enable FaceDetailer for sharper faces (adds ~30-60s)")


@router.post("/generate-raw-hires")
async def generate_image_raw_hires(request: RawHiResRequest):
    """
    Generate a high-resolution image with EXACT prompt + HiRes upscaling.

    This mimics A1111's HiRes Fix workflow:
    1. Generate base image at specified size
    2. Upscale 4x with ESRGAN (4x-UltraSharp)
    3. Scale down to target size (base * hires_scale)
    4. Run second KSampler pass with low denoise for refinement

    Default settings match the reference:
    - Base: 640x1024 → Final: 960x1536 (1.5x)
    - HiRes steps: 10
    - HiRes denoise: 0.3
    """
    from services.image_generation import ImageGenerationService
    from uuid import uuid4
    import random

    try:
        image_service = ImageGenerationService()

        # Use provided seed or generate random
        seed = request.seed or random.randint(0, 2**32 - 1)

        # Determine model to use
        model = request.model or image_service.DEFAULT_MODEL

        # Determine LoRAs to use
        loras_to_use = request.custom_loras

        # Validate custom LoRAs if provided
        if loras_to_use:
            from config.lora_library import get_lora_library
            library = get_lora_library()
            validated_loras = []
            for lora in loras_to_use:
                lora_name = lora.get('name', '')
                lora_weight = lora.get('weight', 1.0)

                # Check if LoRA exists in library
                lora_info = library.get_by_filename(lora_name)
                if lora_info:
                    # Clamp weight to valid range
                    min_weight = lora_info.get('min_weight', 0.0)
                    max_weight = lora_info.get('max_weight', 2.0)
                    lora_weight = max(min_weight, min(max_weight, lora_weight))
                else:
                    # Check if file exists on disk even if not in library
                    if not library.is_available(lora_name):
                        raise HTTPException(
                            status_code=400,
                            detail=f"LoRA '{lora_name}' not found on disk"
                        )
                    # Default weight bounds for unlisted LoRAs
                    lora_weight = max(0.0, min(2.0, lora_weight))

                validated_loras.append({'name': lora_name, 'weight': lora_weight})
            loras_to_use = validated_loras

        if loras_to_use is None and request.use_standard_lora and not request.use_default_loras:
            # Use standard LoRA chain (Youkoso Elf + Breast Slider)
            loras_to_use = image_service.STANDARD_LORAS

        # Determine VAE to use
        vae_model = "sdxl_vae_fp16_fix.safetensors" if request.use_fp16_vae else None

        # Build HiRes workflow with exact prompt
        workflow = image_service._build_hires_workflow(
            positive=request.positive_prompt,
            negative=request.negative_prompt,
            width=request.width,
            height=request.height,
            seed=seed,
            model=model,
            sampler=request.sampler,
            steps=request.steps,
            cfg=request.cfg,
            scheduler=request.scheduler,
            use_loras=request.use_default_loras,
            hires_scale=request.hires_scale,
            hires_steps=request.hires_steps,
            hires_denoise=request.hires_denoise,
            upscaler_model=request.upscaler,
            custom_loras=loras_to_use,
            vae_model=vae_model,
            use_face_detailer=request.use_face_detailer,
        )

        # Execute generation (HiRes takes longer)
        image_data = await image_service._execute_workflow(workflow, timeout=600)

        # Calculate final dimensions
        final_width = int(request.width * request.hires_scale)
        final_height = int(request.height * request.hires_scale)

        # Save to filesystem
        story_uuid = uuid4()
        entity_uuid = uuid4()
        output_dir = image_service.output_base / str(story_uuid) / "raw_hires"
        output_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hires_{seed}_{timestamp}.png"
        file_path = output_dir / filename

        with open(file_path, "wb") as f:
            f.write(image_data)

        # Store settings for history
        generation_settings = {
            "base_width": request.width,
            "base_height": request.height,
            "final_width": final_width,
            "final_height": final_height,
            "steps": request.steps,
            "cfg": request.cfg,
            "sampler": request.sampler,
            "scheduler": request.scheduler,
            "hires_scale": request.hires_scale,
            "hires_steps": request.hires_steps,
            "hires_denoise": request.hires_denoise,
            "upscaler": request.upscaler,
            "negative_prompt": request.negative_prompt,
        }

        # Upload to B2 if available
        file_url = None
        if image_service._b2_storage:
            try:
                b2_result = image_service._b2_storage.upload_image(
                    file_data=image_data,
                    story_id=str(story_uuid),
                    entity_id=str(entity_uuid),
                    image_type="raw_hires",
                    style="raw",
                    extension="png"
                )
                file_url = b2_result["file_url"]
            except Exception as e:
                print(f"[ImageGen] B2 upload failed: {e}")

        # Add to history for gallery
        add_to_history(
            filename=filename,
            url=file_url or f"/images/{filename}",
            local_path=str(file_path),
            seed=seed,
            prompt=request.positive_prompt,
            settings=generation_settings,
        )

        return {
            "success": True,
            "file_path": str(file_path),
            "file_url": file_url,
            "prompt_used": request.positive_prompt,
            "negative_prompt": request.negative_prompt,
            "seed": seed,
            "model": image_service.DEFAULT_MODEL,
            "settings": generation_settings,
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ============================================
# IMAGE HISTORY / GALLERY
# ============================================

class ImageHistoryItem(BaseModel):
    """Single image in the history."""
    filename: str
    url: str
    local_path: str
    created_at: str
    seed: Optional[int] = None
    prompt: Optional[str] = None
    settings: Optional[dict] = None


class ImageHistoryResponse(BaseModel):
    """Response with image history."""
    images: list[ImageHistoryItem]
    total: int


# In-memory storage for image metadata (persists during server runtime)
# In production, this would be stored in the database
_image_history: list[dict] = []


def add_to_history(filename: str, url: str, local_path: str, seed: int, prompt: str, settings: dict):
    """Add an image to the history."""
    _image_history.insert(0, {
        "filename": filename,
        "url": url,
        "local_path": local_path,
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "prompt": prompt,
        "settings": settings,
    })
    # Keep only last 50 images in memory
    if len(_image_history) > 50:
        _image_history.pop()


@router.get("/history", response_model=ImageHistoryResponse)
async def get_image_history(limit: int = Query(default=20, le=50)):
    """
    Get the history of generated images.
    Returns metadata including seed and prompt for each image.
    """
    # Also scan the filesystem for images not in memory
    import os
    from pathlib import Path

    output_dir = Path("/tmp/story-images")
    filesystem_images = []

    if output_dir.exists():
        for img_file in sorted(output_dir.glob("*.png"), key=os.path.getmtime, reverse=True)[:limit]:
            # Check if already in memory history
            if not any(h["filename"] == img_file.name for h in _image_history):
                filesystem_images.append({
                    "filename": img_file.name,
                    "url": f"/images/{img_file.name}",
                    "local_path": str(img_file),
                    "created_at": datetime.fromtimestamp(img_file.stat().st_mtime).isoformat(),
                    "seed": None,  # Unknown for filesystem-only images
                    "prompt": None,
                    "settings": None,
                })

    # Combine memory history with filesystem images
    all_images = _image_history + filesystem_images
    all_images.sort(key=lambda x: x["created_at"], reverse=True)

    return ImageHistoryResponse(
        images=[ImageHistoryItem(**img) for img in all_images[:limit]],
        total=len(all_images),
    )


@router.delete("/history/{filename}")
async def delete_image(filename: str):
    """Delete an image from history and filesystem."""
    import os
    from pathlib import Path

    # Remove from memory history
    global _image_history
    _image_history = [h for h in _image_history if h["filename"] != filename]

    # Remove from filesystem
    file_path = Path("/tmp/story-images") / filename
    if file_path.exists():
        os.remove(file_path)
        return {"success": True, "message": f"Deleted {filename}"}

    return {"success": False, "message": f"File {filename} not found"}


# ============================================
# PROMPT SUGGESTION (Grok-powered)
# ============================================

class PromptSuggestionRequest(BaseModel):
    """Request for AI-generated prompt suggestions."""
    character_name: str = Field(..., description="Name of the character")
    character_description: str = Field(..., description="Full description of the character")
    character_facts: list[str] = Field(default_factory=list, description="Known facts about the character")
    story_premise: Optional[str] = Field(default=None, description="Story premise/setting for context")
    recent_events: Optional[str] = Field(default=None, description="Recent narrative events")
    pose_hint: Optional[str] = Field(default=None, description="User's preferred pose")
    expression_hint: Optional[str] = Field(default=None, description="User's preferred expression")
    is_nsfw: bool = Field(default=True, description="Whether to allow NSFW content")


class PromptSuggestionResponse(BaseModel):
    """Response with AI-generated prompt suggestions."""
    positive_prompt: str
    negative_prompt: str
    recommended_style: str
    recommended_pose: Optional[str] = None
    recommended_expression: Optional[str] = None
    rationale: str


@router.post("/suggest-prompt", response_model=PromptSuggestionResponse)
async def suggest_prompt(request: PromptSuggestionRequest):
    """
    Use Grok to generate optimized Stable Diffusion prompts based on character context.

    This analyzes the character description, story context, and user preferences
    to generate prompts tailored to the Illustrious/Pony model ecosystem.
    """
    try:
        from services.prompt_suggestion import get_prompt_service

        prompt_service = get_prompt_service()
        suggestion = prompt_service.suggest_character_prompt(
            character_name=request.character_name,
            character_description=request.character_description,
            character_facts=request.character_facts,
            story_premise=request.story_premise,
            recent_events=request.recent_events,
            user_pose_hint=request.pose_hint,
            user_expression_hint=request.expression_hint,
            is_nsfw=request.is_nsfw,
        )

        return PromptSuggestionResponse(
            positive_prompt=suggestion.positive_prompt,
            negative_prompt=suggestion.negative_prompt,
            recommended_style=suggestion.recommended_style,
            recommended_pose=suggestion.recommended_pose,
            recommended_expression=suggestion.recommended_expression,
            rationale=suggestion.rationale,
        )

    except Exception as e:
        print(f"[PromptSuggestion] Error: {e}")
        # Return a reasonable default if Grok fails
        return PromptSuggestionResponse(
            positive_prompt=f"masterpiece, best quality, {request.character_name}, {request.character_description[:200]}, detailed face, detailed eyes",
            negative_prompt="worst quality, low quality, bad anatomy, blurry",
            recommended_style="anime",
            recommended_pose=request.pose_hint,
            recommended_expression=request.expression_hint,
            rationale=f"Fallback prompt (Grok unavailable: {str(e)[:100]})",
        )


@router.post("/entities/{entity_id}/suggest-prompt", response_model=PromptSuggestionResponse)
async def suggest_prompt_for_entity(
    entity_id: UUID,
    pose_hint: Optional[str] = None,
    expression_hint: Optional[str] = None,
    is_nsfw: bool = True,
):
    """
    Generate optimized prompts for an existing entity using stored data.

    Fetches the entity's description and facts from the database,
    then uses Grok to generate context-aware prompts.
    """
    try:
        from services.prompt_suggestion import get_prompt_service
        from config.database import supabase

        # Fetch entity data
        entity_result = supabase.table("entities").select("*").eq("id", str(entity_id)).single().execute()
        if not entity_result.data:
            raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

        entity = entity_result.data

        # Get story context if available
        story_premise = None
        if entity.get("story_id"):
            story_result = supabase.table("stories").select("premise").eq("id", entity["story_id"]).single().execute()
            if story_result.data:
                story_premise = story_result.data.get("premise")

        # Get entity facts
        facts_result = supabase.table("entity_facts").select("fact_text").eq("entity_id", str(entity_id)).execute()
        facts = [f["fact_text"] for f in (facts_result.data or [])]

        prompt_service = get_prompt_service()
        suggestion = prompt_service.suggest_character_prompt(
            character_name=entity.get("canonical_name", "Unknown"),
            character_description=entity.get("description", ""),
            character_facts=facts,
            story_premise=story_premise,
            recent_events=None,  # Could fetch from session_memories
            user_pose_hint=pose_hint,
            user_expression_hint=expression_hint,
            is_nsfw=is_nsfw,
        )

        return PromptSuggestionResponse(
            positive_prompt=suggestion.positive_prompt,
            negative_prompt=suggestion.negative_prompt,
            recommended_style=suggestion.recommended_style,
            recommended_pose=suggestion.recommended_pose,
            recommended_expression=suggestion.recommended_expression,
            rationale=suggestion.rationale,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[PromptSuggestion] Error for entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# LORA LIBRARY API
# ============================================

class LoRADetectRequest(BaseModel):
    """Request to detect suggested LoRAs from a prompt."""
    prompt: str = Field(..., description="The prompt to analyze for LoRA suggestions")
    model: str = Field(default="illustrious", description="Target model (illustrious, pony, sdxl)")


class LoRADetectResponse(BaseModel):
    """Response with suggested LoRAs."""
    suggested: list[dict] = Field(description="Suggested LoRAs with match scores")
    auto_enabled: list[dict] = Field(description="LoRAs that should be auto-enabled")


@router.get("/loras")
async def list_loras(
    category: Optional[str] = Query(default=None, description="Filter by category"),
    available_only: bool = Query(default=True, description="Only show LoRAs available on disk"),
):
    """
    List all LoRAs in the library.

    Optionally filter by category (style, body, pose, effect, quality, clothing).
    Returns availability status for each LoRA.
    """
    from config.lora_library import get_lora_library

    library = get_lora_library()

    if category:
        loras = library.get_by_category(category)
    else:
        loras = library.get_all()

    if available_only:
        loras = [l for l in loras if l.get("available", False)]

    return {
        "loras": loras,
        "total": len(loras),
        "categories": library.get_categories(),
    }


@router.get("/loras/categories")
async def list_lora_categories():
    """Get all LoRA categories."""
    from config.lora_library import get_lora_library

    library = get_lora_library()
    categories = library.get_categories()

    # Get count per category
    category_info = []
    for cat in categories:
        loras = library.get_by_category(cat)
        available_count = sum(1 for l in loras if l.get("available", False))
        category_info.append({
            "name": cat,
            "total": len(loras),
            "available": available_count,
        })

    return {"categories": category_info}


@router.post("/loras/detect", response_model=LoRADetectResponse)
async def detect_loras(request: LoRADetectRequest):
    """
    Analyze a prompt and suggest relevant LoRAs.

    Returns:
    - suggested: All LoRAs that match keywords in the prompt, sorted by relevance
    - auto_enabled: LoRAs that should be automatically added (have auto_enable=True)
    """
    from config.lora_library import get_lora_library

    library = get_lora_library()

    suggested = library.detect_from_prompt(request.prompt, request.model)
    auto_enabled = library.get_auto_loras(request.prompt, request.model)

    return LoRADetectResponse(
        suggested=suggested,
        auto_enabled=auto_enabled,
    )


@router.get("/loras/{filename}")
async def get_lora_details(filename: str):
    """Get detailed information about a specific LoRA."""
    from config.lora_library import get_lora_library

    library = get_lora_library()
    lora = library.get_by_filename(filename)

    if not lora:
        raise HTTPException(status_code=404, detail=f"LoRA '{filename}' not found in library")

    return lora


# ============================================
# ENHANCED SCENE GENERATION (with Physical State → LoRA mapping)
# ============================================

class EnhancedSceneRequest(BaseModel):
    """Request for enhanced scene generation with physical state integration."""
    description: str = Field(description="Scene description")
    characters_present: Optional[list[str]] = Field(default=None, description="Names of characters in scene")
    physical_states: Optional[list[dict]] = Field(
        default=None,
        description="Physical states per character: [{character: str, clothing: [], position: str, temporary_states: []}]"
    )
    mood: Optional[str] = Field(default=None, description="Scene mood: intimate, action, romantic, mysterious, tense")
    session_id: Optional[UUID] = Field(default=None)
    use_character_references: bool = Field(default=True, description="Use IP-Adapter with character portraits")
    width: int = Field(default=1024, ge=256, le=1920)
    height: int = Field(default=576, ge=256, le=1080)
    seed: Optional[int] = Field(default=None)
    wait_for_result: bool = Field(default=True, description="If False, returns immediately with task_id")


class EnhancedSceneResponse(BaseModel):
    """Response from enhanced scene generation."""
    task_id: Optional[str] = None
    status: str
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    generation_prompt: Optional[str] = None
    seed: Optional[int] = None
    auto_loras_applied: Optional[list[dict]] = None
    cfg_used: Optional[float] = None
    character_references_used: Optional[dict] = None
    mood: Optional[str] = None
    error: Optional[str] = None


@router.post("/stories/{story_id}/scene/enhanced", response_model=EnhancedSceneResponse)
async def generate_enhanced_scene(
    story_id: UUID,
    request: EnhancedSceneRequest,
):
    """
    Generate a scene with enhanced integrations:

    1. **Physical State → LoRA Mapping**: Automatically selects LoRAs based on
       character physical states (clothing, position, temporary states).

    2. **Character Consistency**: Uses IP-Adapter with character portraits
       to maintain visual consistency across scenes.

    3. **Async Generation**: Can return immediately with task_id for
       non-blocking generation (set wait_for_result=False).

    Physical states should include:
    - clothing: list of clothing items (e.g., ["bikini", "see-through"])
    - position: character position (e.g., "cowgirl", "mating press")
    - temporary_states: temporary conditions (e.g., ["aroused", "sweaty"])

    The service will:
    - Detect appropriate LoRAs from physical states
    - Adjust CFG based on scene mood
    - Enhance the prompt with state details
    - Use character references for IP-Adapter consistency (if available)
    """
    from services.scene_generation import get_scene_generation_service

    db = get_supabase_client()

    # Verify story exists
    story_result = db.table("stories").select("id").eq(
        "id", str(story_id)
    ).execute()

    if not story_result.data:
        raise HTTPException(status_code=404, detail=f"Story {story_id} not found")

    try:
        scene_service = get_scene_generation_service(db_client=db)

        if request.wait_for_result:
            # Synchronous generation
            result = await scene_service.generate_scene_with_characters(
                story_id=story_id,
                scene_description=request.description,
                characters_present=request.characters_present,
                physical_states=request.physical_states,
                mood=request.mood,
                session_id=request.session_id,
                use_character_references=request.use_character_references,
                width=request.width,
                height=request.height,
                seed=request.seed,
            )

            return EnhancedSceneResponse(
                status="completed",
                file_path=result.get("file_path"),
                file_url=result.get("file_url"),
                generation_prompt=result.get("generation_prompt"),
                seed=result.get("seed"),
                auto_loras_applied=result.get("auto_loras_applied"),
                cfg_used=result.get("cfg_used"),
                character_references_used=result.get("character_references_used"),
                mood=result.get("mood"),
            )

        else:
            # Async generation - return task_id
            result = await scene_service.generate_scene_async(
                story_id=story_id,
                scene_description=request.description,
                characters_present=request.characters_present,
                physical_states=request.physical_states,
                mood=request.mood,
                session_id=request.session_id,
                wait_for_result=False,
            )

            return EnhancedSceneResponse(
                task_id=result.get("task_id"),
                status=result.get("status", "pending"),
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scene generation failed: {str(e)}")


@router.get("/tasks/{task_id}")
async def get_generation_task_status(task_id: str):
    """
    Get the status of an async scene generation task.

    Returns:
    - task_id: The task identifier
    - status: pending, processing, completed, failed
    - created_at: When the task was created
    - started_at: When processing started
    - completed_at: When processing finished
    - result: The generation result (if completed)
    - error: Error message (if failed)
    """
    from services.scene_generation import get_scene_generation_service

    scene_service = get_scene_generation_service()
    status = scene_service.get_task_status(task_id)

    if not status:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return status


@router.get("/tasks/{task_id}/result")
async def get_generation_task_result(task_id: str):
    """
    Get the result of a completed generation task.

    Returns 404 if task doesn't exist or isn't completed.
    """
    from services.scene_generation import get_scene_generation_service

    scene_service = get_scene_generation_service()
    status = scene_service.get_task_status(task_id)

    if not status:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} is not completed (status: {status['status']})"
        )

    return scene_service.get_task_result(task_id)


# ============================================
# PHYSICAL STATE → LORA MAPPING API
# ============================================

class PhysicalStateLoRARequest(BaseModel):
    """Request to get LoRA suggestions from physical states."""
    physical_states: list[dict] = Field(
        description="Physical states per character: [{character: str, clothing: [], position: str, temporary_states: []}]"
    )
    scene_description: Optional[str] = Field(default=None, description="Scene description for additional keyword detection")
    mood: Optional[str] = Field(default=None, description="Scene mood")


class PhysicalStateLoRAResponse(BaseModel):
    """Response with LoRA suggestions."""
    suggested_loras: list[dict]
    cfg_adjustment: float
    explanation: Optional[str] = None


@router.post("/loras/from-physical-states", response_model=PhysicalStateLoRAResponse)
async def get_loras_from_physical_states(request: PhysicalStateLoRARequest):
    """
    Get LoRA suggestions based on physical states.

    This is the core of the Physical State → LoRA Mapping system.
    Given character physical states (clothing, position, temporary states),
    it returns appropriate LoRAs to apply.

    Example physical_states:
    ```json
    [
        {
            "character": "Mira",
            "clothing": ["bikini", "see-through"],
            "position": "cowgirl",
            "temporary_states": ["aroused", "sweaty"]
        }
    ]
    ```

    Returns:
    - suggested_loras: List of LoRAs with weights to apply
    - cfg_adjustment: CFG scale adjustment based on mood
    - explanation: Why these LoRAs were selected
    """
    from services.scene_generation import PhysicalStateLoRAMapper

    mapper = PhysicalStateLoRAMapper()

    suggested = mapper.map_states_to_loras(
        physical_states=request.physical_states,
        scene_description=request.scene_description,
        mood=request.mood,
    )

    cfg_adj = mapper.get_cfg_adjustment(request.mood)

    # Build explanation
    explanation_parts = []
    for state in request.physical_states:
        char = state.get("character", "Character")
        clothing = state.get("clothing") or []
        position = state.get("position")
        temps = state.get("temporary_states") or []

        parts = []
        if clothing:
            parts.append(f"clothing: {', '.join(clothing) if isinstance(clothing, list) else clothing}")
        if position:
            parts.append(f"position: {position}")
        if temps:
            parts.append(f"states: {', '.join(temps) if isinstance(temps, list) else temps}")

        if parts:
            explanation_parts.append(f"{char}: {'; '.join(parts)}")

    explanation = f"LoRAs selected based on: {' | '.join(explanation_parts)}" if explanation_parts else None

    return PhysicalStateLoRAResponse(
        suggested_loras=suggested,
        cfg_adjustment=cfg_adj,
        explanation=explanation,
    )


# ============================================
# CHARACTER REFERENCES API
# ============================================

@router.get("/stories/{story_id}/character-references")
async def get_character_references(
    story_id: UUID,
    characters: str = Query(..., description="Comma-separated list of character names"),
):
    """
    Get reference image paths for characters (for IP-Adapter).

    Returns a dict mapping character name to their primary portrait URL/path.
    Used to ensure visual consistency when generating scenes with multiple characters.
    """
    from services.scene_generation import get_scene_generation_service

    db = get_supabase_client()

    # Parse character names
    character_names = [c.strip() for c in characters.split(",") if c.strip()]

    if not character_names:
        raise HTTPException(status_code=400, detail="No character names provided")

    scene_service = get_scene_generation_service(db_client=db)

    references = await scene_service.get_character_references(
        story_id=story_id,
        character_names=character_names,
    )

    return {
        "story_id": str(story_id),
        "characters": references,
        "available_count": sum(1 for v in references.values() if v is not None),
        "total_requested": len(character_names),
    }


# ============================================
# AI-POWERED LORA OPTIMIZATION (GROK)
# ============================================

class AIOptimizeRequest(BaseModel):
    """Request for AI-powered LoRA and settings optimization."""
    prompt: str = Field(..., description="The image generation prompt to optimize")
    style: str = Field(default="anime", description="Target style: anime, realistic")
    max_loras: int = Field(default=4, ge=1, le=6, description="Maximum LoRAs to suggest")


class AIOptimizeResponse(BaseModel):
    """Response with AI-optimized generation settings."""
    selected_loras: list[dict] = Field(description="Recommended LoRAs with weights and reasons")
    cfg_scale: float = Field(description="Recommended CFG scale")
    steps: int = Field(description="Recommended generation steps")
    negative_additions: list[str] = Field(description="Tags to add to negative prompt")
    positive_additions: list[str] = Field(description="Tags to enhance the prompt")
    warnings: list[str] = Field(default=[], description="Any concerns or limitations")
    prompt_improvements: Optional[str] = Field(default=None, description="Suggested prompt refinements")


async def get_lora_registry_from_supabase() -> list[dict]:
    """Fetch LoRA registry from Supabase (worldbuilding database)."""
    import httpx
    import os

    # Use the worldbuilding Supabase (separate from story-memory-system's main DB)
    WORLDBUILDING_URL = "https://mntpiewbprdjpgcbzaca.supabase.co"
    WORLDBUILDING_KEY = os.getenv("SUPABASE_KEY")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{WORLDBUILDING_URL}/rest/v1/lora_registry",
            headers={
                "apikey": WORLDBUILDING_KEY,
                "Authorization": f"Bearer {WORLDBUILDING_KEY}",
            },
            params={
                "select": "filename,display_name,category,description,trigger_keywords,default_weight,min_weight,max_weight,conflicts_with,works_well_with,style_compatibility,is_nsfw,requires_solo,min_breast_size,negative_recommendations,positive_recommendations",
                "is_active": "eq.true",
            }
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"[AI Optimize] Failed to fetch LoRA registry: {response.status_code}")
            return []


async def get_grok_system_prompt() -> str:
    """Fetch the Grok system prompt from Supabase."""
    import httpx
    import os

    WORLDBUILDING_URL = "https://mntpiewbprdjpgcbzaca.supabase.co"
    WORLDBUILDING_KEY = os.getenv("SUPABASE_KEY")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{WORLDBUILDING_URL}/rest/v1/ai_optimization_prompts",
            headers={
                "apikey": WORLDBUILDING_KEY,
                "Authorization": f"Bearer {WORLDBUILDING_KEY}",
            },
            params={
                "select": "system_prompt",
                "prompt_type": "eq.lora_selection",
                "is_active": "eq.true",
                "limit": "1",
            }
        )

        if response.status_code == 200 and response.json():
            return response.json()[0]["system_prompt"]
        else:
            # Fallback system prompt
            return """You are an expert AI image generation assistant. Given a prompt, select optimal LoRAs (max 3-4) and settings. Return JSON only with: selected_loras, cfg_scale, steps, negative_additions, positive_additions, warnings, prompt_improvements."""


@router.post("/loras/ai-optimize", response_model=AIOptimizeResponse)
async def ai_optimize_loras(request: AIOptimizeRequest):
    """
    Use Grok AI to analyze a prompt and recommend optimal LoRAs and generation settings.

    This endpoint:
    1. Fetches the LoRA registry from Supabase
    2. Sends the prompt + registry to Grok
    3. Returns AI-recommended LoRAs, weights, and settings

    The AI considers:
    - Keyword matches in the prompt
    - LoRA conflicts and synergies
    - Style compatibility
    - Quality best practices (max 3-4 LoRAs, balanced weights)
    """
    import httpx
    import os
    import json

    # Get LoRA registry
    lora_registry = await get_lora_registry_from_supabase()

    if not lora_registry:
        # Fallback to basic keyword detection if registry unavailable
        from config.lora_library import get_lora_library
        library = get_lora_library()
        suggested, auto_enabled = library.detect_loras(request.prompt)

        return AIOptimizeResponse(
            selected_loras=[{"name": l["name"], "weight": l.get("weight", 0.6), "reason": "keyword match"} for l in auto_enabled[:request.max_loras]],
            cfg_scale=7.0,
            steps=28,
            negative_additions=["bad quality", "worst quality", "low quality"],
            positive_additions=["masterpiece", "best quality"],
            warnings=["LoRA registry unavailable - using basic keyword matching"],
            prompt_improvements=None,
        )

    # Get system prompt
    system_prompt = await get_grok_system_prompt()

    # Replace {lora_registry} placeholder with actual registry data
    # Simplify registry for API call (reduce token usage)
    simplified_registry = [
        {
            "filename": l["filename"],
            "name": l["display_name"],
            "category": l["category"],
            "description": l["description"][:200] if l["description"] else "",  # Truncate long descriptions
            "keywords": l["trigger_keywords"][:5] if l["trigger_keywords"] else [],  # Limit keywords
            "default_weight": l["default_weight"],
            "conflicts": l["conflicts_with"][:3] if l["conflicts_with"] else [],
            "works_with": l["works_well_with"][:3] if l["works_well_with"] else [],
            "style": l["style_compatibility"],
        }
        for l in lora_registry
    ]

    system_prompt = system_prompt.replace("{lora_registry}", json.dumps(simplified_registry, indent=2))

    # Call Grok API
    XAI_API_KEY = os.getenv("XAI_API_KEY")

    if not XAI_API_KEY:
        raise HTTPException(status_code=500, detail="XAI_API_KEY not configured")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {XAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "grok-3-mini",  # Fast, cost-effective model
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Analyze this prompt and suggest optimal LoRAs and settings:\n\n{request.prompt}\n\nStyle: {request.style}\nMax LoRAs: {request.max_loras}"},
                    ],
                    "temperature": 0.3,  # Lower temperature for more consistent results
                    "max_tokens": 1000,
                }
            )

            if response.status_code != 200:
                print(f"[AI Optimize] Grok API error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=502, detail=f"Grok API error: {response.status_code}")

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON response from Grok
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            try:
                parsed = json.loads(content.strip())
            except json.JSONDecodeError as e:
                print(f"[AI Optimize] Failed to parse Grok response: {content}")
                raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")

            return AIOptimizeResponse(
                selected_loras=parsed.get("selected_loras", [])[:request.max_loras],
                cfg_scale=parsed.get("cfg_scale", 7.0),
                steps=parsed.get("steps", 28),
                negative_additions=parsed.get("negative_additions", []),
                positive_additions=parsed.get("positive_additions", []),
                warnings=parsed.get("warnings", []),
                prompt_improvements=parsed.get("prompt_improvements"),
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Grok API timeout")
    except Exception as e:
        print(f"[AI Optimize] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# CIVITAI INTEGRATION
# ============================================

CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY", "eefd029920db4e6111efba5acd9d819f")
CIVITAI_BASE_URL = "https://civitai.com/api/v1"
COMFYUI_LORA_PATH = "/Users/officemac/Projects/ComfyUI/models/loras"


class CivitAISearchRequest(BaseModel):
    """Search CivitAI for LoRAs."""
    query: str = Field(description="Search query")
    limit: int = Field(default=20, ge=1, le=100)
    nsfw: bool = Field(default=True, description="Include NSFW results")
    sort: str = Field(default="Most Downloaded", description="Sort by: Highest Rated, Most Downloaded, Newest")
    base_model: Optional[str] = Field(default=None, description="Filter by base model: Pony, SDXL, etc.")


class CivitAIModel(BaseModel):
    """CivitAI model info."""
    id: int
    name: str
    description: Optional[str] = None
    type: str
    nsfw: bool
    tags: list[str] = []
    download_count: int = 0
    thumbs_up: int = 0
    creator: Optional[str] = None
    preview_url: Optional[str] = None
    download_url: Optional[str] = None
    filename: Optional[str] = None
    base_model: Optional[str] = None
    trained_words: list[str] = []
    file_size_kb: Optional[float] = None


class CivitAISearchResponse(BaseModel):
    """Search results from CivitAI."""
    models: list[CivitAIModel]
    total: int
    has_more: bool


class CivitAIDownloadRequest(BaseModel):
    """Request to download a LoRA from CivitAI."""
    model_id: int = Field(description="CivitAI model ID")
    version_id: Optional[int] = Field(default=None, description="Specific version ID (uses latest if not specified)")
    custom_filename: Optional[str] = Field(default=None, description="Custom filename (optional)")


class CivitAIDownloadResponse(BaseModel):
    """Response after downloading a LoRA."""
    success: bool
    filename: str
    file_path: str
    file_size_mb: float
    model_name: str
    trained_words: list[str]
    added_to_registry: bool


class LocalLoRAEnrichRequest(BaseModel):
    """Request to enrich local LoRAs with CivitAI metadata."""
    filenames: Optional[list[str]] = Field(default=None, description="Specific files to enrich (all if not specified)")


class EnrichedLoRA(BaseModel):
    """Local LoRA enriched with CivitAI data."""
    filename: str
    civitai_id: Optional[int] = None
    civitai_name: Optional[str] = None
    description: Optional[str] = None
    trained_words: list[str] = []
    tags: list[str] = []
    base_model: Optional[str] = None
    matched_by: Optional[str] = None  # "hash" or "name"


class LocalLoRAEnrichResponse(BaseModel):
    """Response with enriched LoRA data."""
    enriched: list[EnrichedLoRA]
    not_found: list[str]
    total_local: int


@router.get("/civitai/search", response_model=CivitAISearchResponse)
async def search_civitai(
    query: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    nsfw: bool = Query(True, description="Include NSFW"),
    sort: str = Query("Most Downloaded", description="Sort order"),
    base_model: Optional[str] = Query(None, description="Filter by base model"),
):
    """
    Search CivitAI for LoRA models.
    """
    params = {
        "query": query,
        "limit": limit,
        "types": "LORA",
        "nsfw": str(nsfw).lower(),
        "sort": sort,
    }
    if base_model:
        params["baseModels"] = base_model

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{CIVITAI_BASE_URL}/models",
            params=params,
            headers={"Authorization": f"Bearer {CIVITAI_API_KEY}"}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="CivitAI API error")

        data = response.json()
        items = data.get("items", [])

        models = []
        for item in items:
            # Get the latest version info
            versions = item.get("modelVersions", [])
            latest = versions[0] if versions else {}
            files = latest.get("files", [])
            primary_file = next((f for f in files if f.get("primary")), files[0] if files else {})
            images = latest.get("images", [])

            models.append(CivitAIModel(
                id=item["id"],
                name=item["name"],
                description=item.get("description", "")[:500] if item.get("description") else None,
                type=item.get("type", "LORA"),
                nsfw=item.get("nsfw", False),
                tags=item.get("tags", []),
                download_count=item.get("stats", {}).get("downloadCount", 0),
                thumbs_up=item.get("stats", {}).get("thumbsUpCount", 0),
                creator=item.get("creator", {}).get("username"),
                preview_url=images[0].get("url") if images else None,
                download_url=latest.get("downloadUrl"),
                filename=primary_file.get("name"),
                base_model=latest.get("baseModel"),
                trained_words=latest.get("trainedWords", []),
                file_size_kb=primary_file.get("sizeKB"),
            ))

        return CivitAISearchResponse(
            models=models,
            total=len(models),
            has_more=len(items) == limit,
        )


@router.get("/civitai/model/{model_id}")
async def get_civitai_model(model_id: int):
    """
    Get detailed info for a specific CivitAI model.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{CIVITAI_BASE_URL}/models/{model_id}",
            headers={"Authorization": f"Bearer {CIVITAI_API_KEY}"}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Model not found")

        return response.json()


@router.post("/civitai/download", response_model=CivitAIDownloadResponse)
async def download_civitai_lora(request: CivitAIDownloadRequest, background_tasks: BackgroundTasks):
    """
    Download a LoRA from CivitAI and install it to ComfyUI.
    """
    import aiofiles
    from pathlib import Path

    # First get model info
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{CIVITAI_BASE_URL}/models/{request.model_id}",
            headers={"Authorization": f"Bearer {CIVITAI_API_KEY}"}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Model not found on CivitAI")

        model_data = response.json()
        model_name = model_data["name"]
        versions = model_data.get("modelVersions", [])

        if not versions:
            raise HTTPException(status_code=404, detail="No versions available")

        # Find the requested version or use latest
        if request.version_id:
            version = next((v for v in versions if v["id"] == request.version_id), None)
            if not version:
                raise HTTPException(status_code=404, detail="Version not found")
        else:
            version = versions[0]

        files = version.get("files", [])
        primary_file = next((f for f in files if f.get("primary")), files[0] if files else None)

        if not primary_file:
            raise HTTPException(status_code=404, detail="No downloadable file found")

        download_url = version.get("downloadUrl")
        original_filename = primary_file.get("name", f"{model_name}.safetensors")
        filename = request.custom_filename or original_filename
        file_size_kb = primary_file.get("sizeKB", 0)
        trained_words = version.get("trainedWords", [])
        base_model = version.get("baseModel")

        # Ensure filename ends with .safetensors
        if not filename.endswith(".safetensors"):
            filename += ".safetensors"

        file_path = Path(COMFYUI_LORA_PATH) / filename

        # Check if already exists
        if file_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"LoRA already exists: {filename}"
            )

        # Download the file
        print(f"[CivitAI] Downloading {model_name} to {file_path}...")

        async with httpx.AsyncClient(timeout=300, follow_redirects=True) as download_client:
            download_response = await download_client.get(
                download_url,
                headers={"Authorization": f"Bearer {CIVITAI_API_KEY}"}
            )

            if download_response.status_code != 200:
                raise HTTPException(
                    status_code=download_response.status_code,
                    detail="Failed to download from CivitAI"
                )

            # Write to file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(download_response.content)

        actual_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"[CivitAI] Downloaded {filename} ({actual_size_mb:.1f} MB)")

        # Add to registry in background
        added_to_registry = False
        try:
            db = get_supabase_client()
            # Use worldbuilding database for lora_registry
            from supabase import create_client
            worldbuilding_url = "https://mntpiewbprdjpgcbzaca.supabase.co"
            worldbuilding_key = os.getenv("SUPABASE_KEY", "")
            wb_client = create_client(worldbuilding_url, worldbuilding_key)

            # Determine category from tags
            category = "style"  # default
            tag_lower = [t.lower() for t in model_data.get("tags", [])]
            if any(t in tag_lower for t in ["body", "breast", "ass", "thick"]):
                category = "body"
            elif any(t in tag_lower for t in ["pose", "position"]):
                category = "pose"
            elif any(t in tag_lower for t in ["clothing", "outfit", "lingerie"]):
                category = "clothing"
            elif any(t in tag_lower for t in ["effect", "lighting"]):
                category = "effect"

            # Clean description (strip HTML)
            import re
            description = model_data.get("description", "")
            if description:
                description = re.sub(r'<[^>]+>', '', description)[:500]

            wb_client.table("lora_registry").upsert({
                "filename": filename,
                "display_name": model_name,
                "category": category,
                "description": description or f"Downloaded from CivitAI: {model_name}",
                "trigger_keywords": trained_words,
                "default_weight": 0.7,
                "style_compatibility": [base_model] if base_model else ["Pony"],
                "is_nsfw": model_data.get("nsfw", True),
                "civitai_id": request.model_id,
                "civitai_version_id": version.get("id"),
            }, on_conflict="filename").execute()

            added_to_registry = True
            print(f"[CivitAI] Added {filename} to registry")
        except Exception as e:
            print(f"[CivitAI] Failed to add to registry: {e}")

        return CivitAIDownloadResponse(
            success=True,
            filename=filename,
            file_path=str(file_path),
            file_size_mb=actual_size_mb,
            model_name=model_name,
            trained_words=trained_words,
            added_to_registry=added_to_registry,
        )


@router.post("/civitai/enrich-local", response_model=LocalLoRAEnrichResponse)
async def enrich_local_loras(request: LocalLoRAEnrichRequest):
    """
    Enrich local LoRAs with metadata from CivitAI by matching file hashes or names.
    """
    import hashlib
    from pathlib import Path

    lora_path = Path(COMFYUI_LORA_PATH)
    if not lora_path.exists():
        raise HTTPException(status_code=404, detail=f"LoRA directory not found: {COMFYUI_LORA_PATH}")

    # Get list of local LoRA files
    if request.filenames:
        local_files = [lora_path / f for f in request.filenames if (lora_path / f).exists()]
    else:
        local_files = list(lora_path.glob("*.safetensors"))

    enriched = []
    not_found = []

    async with httpx.AsyncClient(timeout=30) as client:
        for file_path in local_files:
            filename = file_path.name
            print(f"[CivitAI Enrich] Processing {filename}...")

            # Try to match by hash first (most accurate)
            matched = False
            try:
                # Calculate full SHA256 hash of the entire file
                # CivitAI accepts both AutoV2 (first 10 chars) and full SHA256
                sha256_hash = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    # Read in chunks to handle large files efficiently
                    for chunk in iter(lambda: f.read(8192), b''):
                        sha256_hash.update(chunk)
                full_sha256 = sha256_hash.hexdigest().upper()

                # Also calculate AutoV2 hash as fallback (first 256KB, first 10 chars)
                with open(file_path, 'rb') as f:
                    autov2_content = f.read(256 * 1024)
                    autov2_hash = hashlib.sha256(autov2_content).hexdigest().upper()[:10]

                print(f"[CivitAI Enrich] {filename}: SHA256={full_sha256[:16]}..., AutoV2={autov2_hash}")

                # Try full SHA256 first
                response = await client.get(
                    f"{CIVITAI_BASE_URL}/model-versions/by-hash/{full_sha256}",
                    headers={"Authorization": f"Bearer {CIVITAI_API_KEY}"}
                )

                # If full SHA256 fails, try AutoV2 hash
                if response.status_code != 200:
                    print(f"[CivitAI Enrich] Full SHA256 not found, trying AutoV2...")
                    response = await client.get(
                        f"{CIVITAI_BASE_URL}/model-versions/by-hash/{autov2_hash}",
                        headers={"Authorization": f"Bearer {CIVITAI_API_KEY}"}
                    )

                if response.status_code == 200:
                    version_data = response.json()
                    model_id = version_data.get("modelId")

                    # Get full model info
                    model_response = await client.get(
                        f"{CIVITAI_BASE_URL}/models/{model_id}",
                        headers={"Authorization": f"Bearer {CIVITAI_API_KEY}"}
                    )

                    if model_response.status_code == 200:
                        model_data = model_response.json()
                        import re
                        description = model_data.get("description", "")
                        if description:
                            description = re.sub(r'<[^>]+>', '', description)[:500]

                        enriched.append(EnrichedLoRA(
                            filename=filename,
                            civitai_id=model_id,
                            civitai_name=model_data.get("name"),
                            description=description,
                            trained_words=version_data.get("trainedWords", []),
                            tags=model_data.get("tags", []),
                            base_model=version_data.get("baseModel"),
                            matched_by="hash",
                        ))
                        matched = True
                        print(f"[CivitAI Enrich] Matched {filename} by hash -> {model_data.get('name')}")

            except Exception as e:
                print(f"[CivitAI Enrich] Hash lookup failed for {filename}: {e}")

            # If hash didn't match, try name search
            if not matched:
                try:
                    # Clean up filename for search
                    search_name = filename.replace(".safetensors", "").replace("_", " ").replace("-", " ")
                    # Remove common suffixes
                    for suffix in ["pony", "v1", "v2", "v3", "v4", "v5", "xl", "sdxl"]:
                        search_name = search_name.lower().replace(suffix, "").strip()

                    response = await client.get(
                        f"{CIVITAI_BASE_URL}/models",
                        params={
                            "query": search_name[:50],
                            "types": "LORA",
                            "limit": 5,
                            "nsfw": "true",
                        },
                        headers={"Authorization": f"Bearer {CIVITAI_API_KEY}"}
                    )

                    if response.status_code == 200:
                        data = response.json()
                        items = data.get("items", [])

                        # Find best match
                        for item in items:
                            item_name = item.get("name", "").lower()
                            if search_name.lower() in item_name or item_name in search_name.lower():
                                versions = item.get("modelVersions", [])
                                latest = versions[0] if versions else {}

                                import re
                                description = item.get("description", "")
                                if description:
                                    description = re.sub(r'<[^>]+>', '', description)[:500]

                                enriched.append(EnrichedLoRA(
                                    filename=filename,
                                    civitai_id=item["id"],
                                    civitai_name=item.get("name"),
                                    description=description,
                                    trained_words=latest.get("trainedWords", []),
                                    tags=item.get("tags", []),
                                    base_model=latest.get("baseModel"),
                                    matched_by="name",
                                ))
                                matched = True
                                print(f"[CivitAI Enrich] Matched {filename} by name -> {item.get('name')}")
                                break

                except Exception as e:
                    print(f"[CivitAI Enrich] Name search failed for {filename}: {e}")

            if not matched:
                not_found.append(filename)
                enriched.append(EnrichedLoRA(
                    filename=filename,
                    matched_by=None,
                ))

    return LocalLoRAEnrichResponse(
        enriched=enriched,
        not_found=not_found,
        total_local=len(local_files),
    )


@router.post("/civitai/sync-registry")
async def sync_civitai_to_registry():
    """
    Sync all enriched CivitAI data to the Supabase lora_registry.
    """
    # First enrich all local LoRAs
    enrich_response = await enrich_local_loras(LocalLoRAEnrichRequest())

    # Connect to worldbuilding database
    from supabase import create_client
    worldbuilding_url = "https://mntpiewbprdjpgcbzaca.supabase.co"
    worldbuilding_key = os.getenv("SUPABASE_KEY", "")
    wb_client = create_client(worldbuilding_url, worldbuilding_key)

    updated = 0
    for lora in enrich_response.enriched:
        if lora.civitai_id:
            try:
                # Determine category from tags
                category = "style"
                tag_lower = [t.lower() for t in lora.tags]
                if any(t in tag_lower for t in ["body", "breast", "ass", "thick", "oppai"]):
                    category = "body"
                elif any(t in tag_lower for t in ["pose", "position"]):
                    category = "pose"
                elif any(t in tag_lower for t in ["clothing", "outfit", "lingerie"]):
                    category = "clothing"
                elif any(t in tag_lower for t in ["effect", "lighting"]):
                    category = "effect"

                wb_client.table("lora_registry").upsert({
                    "filename": lora.filename,
                    "display_name": lora.civitai_name or lora.filename.replace(".safetensors", ""),
                    "category": category,
                    "description": lora.description or f"LoRA: {lora.civitai_name}",
                    "trigger_keywords": lora.trained_words,
                    "default_weight": 0.7,
                    "style_compatibility": [lora.base_model] if lora.base_model else ["Pony"],
                    "civitai_id": lora.civitai_id,
                }, on_conflict="filename").execute()

                updated += 1
            except Exception as e:
                print(f"[CivitAI Sync] Failed to update {lora.filename}: {e}")

    return {
        "total_local": enrich_response.total_local,
        "enriched_count": len([l for l in enrich_response.enriched if l.civitai_id]),
        "updated_in_registry": updated,
        "not_found_on_civitai": enrich_response.not_found,
    }