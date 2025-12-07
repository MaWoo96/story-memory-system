"""
Image generation API endpoints.

Handles character portrait generation, scene illustrations, and image management.
"""

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
