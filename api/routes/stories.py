"""
Stories API endpoints.
Currently: entity image generation.
"""

from fastapi import APIRouter, Depends, HTTPException, Path, Body
from uuid import UUID
from typing import Optional

# Assume these imports exist based on project structure
from services.image_gen import ImageGenService
from services.storage import StorageService
from schemas.extraction import Entity
from api.dependencies import get_storage_service

router = APIRouter(prefix="/stories", tags=["stories"])

@router.post(
    "{story_id}/entities/{entity_id}/image",
    response_model=dict
)
async def generate_entity_image(
    story_id: UUID = Path(..., description="ID of the story"),
    entity_id: UUID = Path(..., description="ID of the entity"),
    storage: StorageService = Depends(get_storage_service),
    image_gen: ImageGenService = Depends(ImageGenService)
):
    """
    Generate NSFW-safe image for the specified story entity using Stable Diffusion.
    Saves image to /tmp/story-images/{story_id}/{entity_name}.png and returns path.
    """
    entity: Optional[Entity] = await storage.get_entity(story_id, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    image_path = image_gen.generate_entity_image(entity.dict(), str(story_id))
    return {"url": image_path}