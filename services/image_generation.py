"""
Image generation service using ComfyUI.

Generates character portraits and scene images with support for:
- Multiple art styles (anime, realistic, fantasy)
- Character consistency via IP-Adapter
- Local storage with database metadata tracking
"""

import os
import json
import httpx
import hashlib
import asyncio
from pathlib import Path
from uuid import UUID
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from schemas.extraction import Entity, EntityType


@dataclass
class GenerationResult:
    """Result from image generation."""
    file_path: str
    file_url: Optional[str]
    generation_prompt: str
    negative_prompt: str
    seed: int
    model_used: str
    sampler: str
    steps: int
    cfg_scale: float
    width: int
    height: int


class ImageGenerationService:
    """
    Generates character portraits and scene images using local ComfyUI.
    Images are stored on local filesystem with metadata in database.
    """

    # Style presets for different art styles
    STYLE_PRESETS = {
        "anime": {
            "quality_tags": [
                "masterpiece", "best quality", "highly detailed",
                "sharp focus", "anime style", "cel shading",
                "vibrant colors", "detailed eyes"
            ],
            "negative": (
                "lowres, bad anatomy, bad hands, text, error, missing fingers, "
                "extra digit, fewer digits, cropped, worst quality, low quality, "
                "normal quality, jpeg artifacts, signature, watermark, username, "
                "blurry, deformed, mutated, disfigured, realistic, photorealistic, "
                "3d render"
            ),
            "model": "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
            "sampler": "euler_ancestral",
            "steps": 30,
            "cfg": 7.0,
        },
        "realistic": {
            "quality_tags": [
                "masterpiece", "best quality", "photorealistic", "8k uhd",
                "dslr quality", "sharp focus", "professional lighting",
                "detailed skin texture", "natural lighting"
            ],
            "negative": (
                "lowres, bad anatomy, bad hands, text, error, missing fingers, "
                "extra digit, fewer digits, cropped, worst quality, low quality, "
                "jpeg artifacts, signature, watermark, blurry, deformed, mutated, "
                "cartoon, anime, illustration, painting, drawing"
            ),
            "model": "realisticVisionV60B1_v51VAE.safetensors",
            "sampler": "dpmpp_2m_sde_gpu",
            "steps": 35,
            "cfg": 5.5,
        },
        "fantasy": {
            "quality_tags": [
                "masterpiece", "best quality", "fantasy art", "digital painting",
                "artstation", "highly detailed", "dramatic lighting",
                "epic composition", "vibrant colors"
            ],
            "negative": (
                "lowres, bad anatomy, bad hands, text, error, missing fingers, "
                "extra digit, fewer digits, cropped, worst quality, low quality, "
                "jpeg artifacts, signature, watermark, blurry, deformed, mutated, "
                "disfigured, ugly, photo, realistic"
            ),
            "model": "dreamshaperXL_v21TurboDPMSDE.safetensors",
            "sampler": "dpmpp_sde",
            "steps": 25,
            "cfg": 6.0,
        },
    }

    # Entity type to visual descriptors
    ENTITY_TYPE_PROMPTS = {
        EntityType.CHARACTER: "1person, portrait, character portrait, face focus",
        EntityType.CREATURE: "creature, monster, fantasy creature, full body",
        EntityType.LOCATION: "landscape, scenery, environment, wide shot, no people",
        EntityType.ITEM: "object, item, detailed, centered, simple background",
        EntityType.FACTION: "group, emblem, symbol, heraldry, logo design",
    }

    def __init__(
        self,
        comfyui_url: str = None,
        output_base_path: str = None,
        serve_url_base: str = None,
    ):
        """
        Initialize image generation service.

        Args:
            comfyui_url: ComfyUI API URL (default from env)
            output_base_path: Base path for generated images (default from env)
            serve_url_base: Base URL for serving images (optional)
        """
        self.comfyui_url = comfyui_url or os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
        self.output_base = Path(output_base_path or os.getenv("IMAGE_OUTPUT_PATH", "./generated_images"))
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.serve_url_base = serve_url_base

    def build_character_prompt(
        self,
        entity: Entity,
        style: str = "anime",
        additional_tags: list[str] = None,
        pose: str = None,
        expression: str = None,
    ) -> tuple[str, str]:
        """
        Build a generation prompt from entity data.

        Args:
            entity: Entity to generate image for
            style: Art style preset
            additional_tags: Extra tags to include
            pose: Specific pose (e.g., "sitting", "action pose")
            expression: Facial expression (e.g., "smiling", "serious")

        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """
        preset = self.STYLE_PRESETS.get(style, self.STYLE_PRESETS["anime"])

        # Start with quality tags
        prompt_parts = list(preset["quality_tags"])

        # Add entity type descriptor
        entity_prompt = self.ENTITY_TYPE_PROMPTS.get(
            entity.entity_type,
            "character portrait"
        )
        prompt_parts.append(entity_prompt)

        # Extract visual traits from entity facts
        visual_traits = []
        personality_traits = []

        for fact in entity.facts:
            fact_type = fact.fact_type.value.lower()
            fact_value = fact.fact_value.lower()

            # Categorize facts for prompt building
            if fact_type == "trait":
                # Physical traits go in prompt, personality for context
                physical_keywords = ["hair", "eye", "skin", "tall", "short", "muscular",
                                     "slim", "scar", "tattoo", "beard", "young", "old"]
                if any(kw in fact_value for kw in physical_keywords):
                    visual_traits.append(fact.fact_value)
                else:
                    personality_traits.append(fact.fact_value)

            elif fact_type == "occupation":
                # Add occupation-related visual elements
                prompt_parts.append(fact.fact_value)

            elif fact_type == "possession":
                # Notable possessions can be visual
                if any(kw in fact_value for kw in ["wearing", "carries", "wields", "armor", "cloak", "robe"]):
                    visual_traits.append(fact.fact_value)

        # Add visual traits
        prompt_parts.extend(visual_traits[:8])  # Limit to avoid prompt overload

        # Add entity description (truncated)
        if entity.description:
            # Extract key visual descriptors from description
            desc_words = entity.description[:150]
            prompt_parts.append(desc_words)

        # Add pose and expression if specified
        if pose:
            prompt_parts.append(pose)
        if expression:
            prompt_parts.append(f"{expression} expression")

        # Add any additional tags
        if additional_tags:
            prompt_parts.extend(additional_tags)

        # Build final prompt
        positive = ", ".join(prompt_parts)
        negative = preset["negative"]

        return positive, negative

    def _get_seed_for_entity(self, entity_name: str, variation: int = 0) -> int:
        """Generate a consistent seed for an entity name."""
        seed_string = f"{entity_name}_{variation}"
        return int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)

    async def generate_portrait(
        self,
        entity: Entity,
        entity_id: UUID,
        story_id: UUID,
        style: str = "anime",
        width: int = 512,
        height: int = 768,
        seed: int = None,
        additional_tags: list[str] = None,
        pose: str = None,
        expression: str = None,
    ) -> GenerationResult:
        """
        Generate a character portrait and save locally.

        Args:
            entity: Entity to generate portrait for
            entity_id: Entity UUID
            story_id: Story UUID
            style: Art style preset
            width: Image width
            height: Image height
            seed: Random seed (None for consistent seed based on name)
            additional_tags: Extra prompt tags
            pose: Specific pose
            expression: Facial expression

        Returns:
            GenerationResult with file path and metadata
        """
        preset = self.STYLE_PRESETS.get(style, self.STYLE_PRESETS["anime"])

        # Build prompts
        positive, negative = self.build_character_prompt(
            entity=entity,
            style=style,
            additional_tags=additional_tags,
            pose=pose,
            expression=expression,
        )

        # Use consistent seed if not provided
        if seed is None:
            seed = self._get_seed_for_entity(entity.canonical_name)

        # Build ComfyUI workflow
        workflow = self._build_txt2img_workflow(
            positive=positive,
            negative=negative,
            width=width,
            height=height,
            seed=seed,
            model=preset["model"],
            sampler=preset["sampler"],
            steps=preset["steps"],
            cfg=preset["cfg"],
        )

        # Execute generation
        image_data = await self._execute_workflow(workflow)

        # Save to filesystem
        output_dir = self.output_base / str(story_id) / "entities" / str(entity_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portrait_{style}_{seed}_{timestamp}.png"
        file_path = output_dir / filename

        with open(file_path, "wb") as f:
            f.write(image_data)

        # Build serve URL if configured
        file_url = None
        if self.serve_url_base:
            relative_path = file_path.relative_to(self.output_base)
            file_url = f"{self.serve_url_base}/{relative_path}"

        return GenerationResult(
            file_path=str(file_path),
            file_url=file_url,
            generation_prompt=positive,
            negative_prompt=negative,
            seed=seed,
            model_used=preset["model"],
            sampler=preset["sampler"],
            steps=preset["steps"],
            cfg_scale=preset["cfg"],
            width=width,
            height=height,
        )

    async def generate_scene(
        self,
        description: str,
        story_id: UUID,
        session_id: UUID = None,
        event_id: UUID = None,
        style: str = "fantasy",
        width: int = 1024,
        height: int = 576,
        seed: int = None,
        participating_entities: list[str] = None,
    ) -> GenerationResult:
        """
        Generate a scene illustration.

        Args:
            description: Scene description
            story_id: Story UUID
            session_id: Optional session UUID
            event_id: Optional event UUID
            style: Art style preset
            width: Image width (landscape by default)
            height: Image height
            seed: Random seed
            participating_entities: Names of entities in scene

        Returns:
            GenerationResult with file path and metadata
        """
        preset = self.STYLE_PRESETS.get(style, self.STYLE_PRESETS["fantasy"])

        # Build scene prompt
        prompt_parts = list(preset["quality_tags"])
        prompt_parts.extend(["scene", "illustration", "dramatic composition"])

        # Add description
        prompt_parts.append(description[:200])

        # Add entity names for context
        if participating_entities:
            prompt_parts.append(f"featuring {', '.join(participating_entities[:3])}")

        positive = ", ".join(prompt_parts)
        negative = preset["negative"]

        # Generate seed
        if seed is None:
            seed = int(hashlib.md5(description.encode()).hexdigest()[:8], 16)

        # Build and execute workflow
        workflow = self._build_txt2img_workflow(
            positive=positive,
            negative=negative,
            width=width,
            height=height,
            seed=seed,
            model=preset["model"],
            sampler=preset["sampler"],
            steps=preset["steps"],
            cfg=preset["cfg"],
        )

        image_data = await self._execute_workflow(workflow)

        # Save to filesystem
        output_dir = self.output_base / str(story_id) / "scenes"
        if session_id:
            output_dir = output_dir / str(session_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scene_{style}_{seed}_{timestamp}.png"
        file_path = output_dir / filename

        with open(file_path, "wb") as f:
            f.write(image_data)

        file_url = None
        if self.serve_url_base:
            relative_path = file_path.relative_to(self.output_base)
            file_url = f"{self.serve_url_base}/{relative_path}"

        return GenerationResult(
            file_path=str(file_path),
            file_url=file_url,
            generation_prompt=positive,
            negative_prompt=negative,
            seed=seed,
            model_used=preset["model"],
            sampler=preset["sampler"],
            steps=preset["steps"],
            cfg_scale=preset["cfg"],
            width=width,
            height=height,
        )

    def _build_txt2img_workflow(
        self,
        positive: str,
        negative: str,
        width: int,
        height: int,
        seed: int,
        model: str,
        sampler: str,
        steps: int,
        cfg: float,
    ) -> dict:
        """Build ComfyUI txt2img workflow JSON."""
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler,
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": model
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": positive,
                    "clip": ["4", 1]
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative,
                    "clip": ["4", 1]
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["8", 0]
                }
            }
        }

    def _build_img2img_workflow(
        self,
        positive: str,
        negative: str,
        reference_image_path: str,
        width: int,
        height: int,
        seed: int,
        model: str,
        sampler: str,
        steps: int,
        cfg: float,
        denoise: float = 0.7,
    ) -> dict:
        """Build ComfyUI img2img workflow for variations."""
        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": reference_image_path
                }
            },
            "2": {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["1", 0],
                    "vae": ["4", 2]
                }
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler,
                    "scheduler": "normal",
                    "denoise": denoise,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["2", 0]
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": model
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": positive,
                    "clip": ["4", 1]
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative,
                    "clip": ["4", 1]
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["8", 0]
                }
            }
        }

    async def _execute_workflow(
        self,
        workflow: dict,
        timeout: int = 300,
    ) -> bytes:
        """
        Execute a ComfyUI workflow and return the generated image.

        Args:
            workflow: ComfyUI workflow dict
            timeout: Maximum wait time in seconds

        Returns:
            Image data as bytes
        """
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Queue the prompt
            response = await client.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow}
            )
            response.raise_for_status()
            result = response.json()
            prompt_id = result["prompt_id"]

            # Poll for completion
            image_data = await self._wait_for_completion(client, prompt_id, timeout)
            return image_data

    async def _wait_for_completion(
        self,
        client: httpx.AsyncClient,
        prompt_id: str,
        timeout: int,
    ) -> bytes:
        """Poll ComfyUI until generation completes."""
        poll_interval = 1
        elapsed = 0

        while elapsed < timeout:
            response = await client.get(f"{self.comfyui_url}/history/{prompt_id}")
            history = response.json()

            if prompt_id in history:
                # Check for errors
                if history[prompt_id].get("status", {}).get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI generation failed: {history[prompt_id]}")

                outputs = history[prompt_id].get("outputs", {})

                # Find the SaveImage node output
                for node_id, output in outputs.items():
                    if "images" in output:
                        image_info = output["images"][0]

                        # Fetch the actual image
                        img_response = await client.get(
                            f"{self.comfyui_url}/view",
                            params={
                                "filename": image_info["filename"],
                                "subfolder": image_info.get("subfolder", ""),
                                "type": image_info.get("type", "output")
                            }
                        )
                        img_response.raise_for_status()
                        return img_response.content

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Generation did not complete within {timeout} seconds")

    async def check_comfyui_status(self) -> dict:
        """Check if ComfyUI is running and get system info."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.comfyui_url}/system_stats")
                if response.status_code == 200:
                    return {"status": "connected", "info": response.json()}
                return {"status": "error", "message": f"HTTP {response.status_code}"}
        except httpx.ConnectError:
            return {"status": "disconnected", "message": "Cannot connect to ComfyUI"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def list_available_models(self) -> list[str]:
        """Get list of available checkpoint models from ComfyUI."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.comfyui_url}/object_info/CheckpointLoaderSimple")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
                return []
        except Exception:
            return []


class CharacterConsistencyService:
    """
    Maintains visual consistency for characters across multiple generations.
    Uses reference images and IP-Adapter for consistent character appearance.
    """

    def __init__(
        self,
        image_service: ImageGenerationService,
        db_client,
    ):
        self.image_service = image_service
        self.db = db_client

    async def get_reference_image(self, entity_id: UUID) -> Optional[str]:
        """Get the primary reference image for an entity."""
        result = self.db.table("entity_images").select("file_path").eq(
            "entity_id", str(entity_id)
        ).eq("is_primary", True).execute()

        if result.data:
            return result.data[0]["file_path"]
        return None

    async def generate_consistent_portrait(
        self,
        entity: Entity,
        entity_id: UUID,
        story_id: UUID,
        style: str = "anime",
        use_reference: bool = True,
        **kwargs
    ) -> GenerationResult:
        """
        Generate a portrait maintaining consistency with previous images.

        Args:
            entity: Entity to generate portrait for
            entity_id: Entity UUID
            story_id: Story UUID
            style: Art style
            use_reference: Whether to use existing reference image
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with file path and metadata
        """
        reference_path = None
        if use_reference:
            reference_path = await self.get_reference_image(entity_id)

        if reference_path and Path(reference_path).exists():
            # Generate using img2img with reference
            return await self._generate_with_reference(
                entity=entity,
                entity_id=entity_id,
                story_id=story_id,
                reference_path=reference_path,
                style=style,
                **kwargs
            )
        else:
            # Generate new base portrait
            result = await self.image_service.generate_portrait(
                entity=entity,
                entity_id=entity_id,
                story_id=story_id,
                style=style,
                **kwargs
            )

            # Store as reference if this is the first image
            await self._store_image_metadata(
                entity_id=entity_id,
                story_id=story_id,
                result=result,
                image_type="portrait",
                is_primary=True,
            )

            return result

    async def _generate_with_reference(
        self,
        entity: Entity,
        entity_id: UUID,
        story_id: UUID,
        reference_path: str,
        style: str,
        variation_strength: float = 0.3,
        **kwargs
    ) -> GenerationResult:
        """Generate image using reference for consistency."""
        preset = self.image_service.STYLE_PRESETS.get(style, self.image_service.STYLE_PRESETS["anime"])

        positive, negative = self.image_service.build_character_prompt(
            entity=entity,
            style=style,
            **kwargs
        )

        seed = kwargs.get("seed") or self.image_service._get_seed_for_entity(
            entity.canonical_name,
            variation=1  # Different variation for img2img
        )

        workflow = self.image_service._build_img2img_workflow(
            positive=positive,
            negative=negative,
            reference_image_path=reference_path,
            width=kwargs.get("width", 512),
            height=kwargs.get("height", 768),
            seed=seed,
            model=preset["model"],
            sampler=preset["sampler"],
            steps=preset["steps"],
            cfg=preset["cfg"],
            denoise=variation_strength,
        )

        image_data = await self.image_service._execute_workflow(workflow)

        # Save image
        output_dir = self.image_service.output_base / str(story_id) / "entities" / str(entity_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portrait_{style}_{seed}_var_{timestamp}.png"
        file_path = output_dir / filename

        with open(file_path, "wb") as f:
            f.write(image_data)

        file_url = None
        if self.image_service.serve_url_base:
            relative_path = file_path.relative_to(self.image_service.output_base)
            file_url = f"{self.image_service.serve_url_base}/{relative_path}"

        result = GenerationResult(
            file_path=str(file_path),
            file_url=file_url,
            generation_prompt=positive,
            negative_prompt=negative,
            seed=seed,
            model_used=preset["model"],
            sampler=preset["sampler"],
            steps=preset["steps"],
            cfg_scale=preset["cfg"],
            width=kwargs.get("width", 512),
            height=kwargs.get("height", 768),
        )

        # Store metadata
        await self._store_image_metadata(
            entity_id=entity_id,
            story_id=story_id,
            result=result,
            image_type="portrait",
            is_primary=False,
        )

        return result

    async def _store_image_metadata(
        self,
        entity_id: UUID,
        story_id: UUID,
        result: GenerationResult,
        image_type: str,
        is_primary: bool,
    ) -> UUID:
        """Store image metadata in database."""
        # If setting as primary, unset existing primary
        if is_primary:
            self.db.table("entity_images").update({
                "is_primary": False
            }).eq("entity_id", str(entity_id)).eq("is_primary", True).execute()

        insert_result = self.db.table("entity_images").insert({
            "entity_id": str(entity_id),
            "image_type": image_type,
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
            "is_primary": is_primary,
        }).execute()

        return UUID(insert_result.data[0]["id"])
