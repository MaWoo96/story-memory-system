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
from services.storage_b2 import get_b2_storage, B2StorageService


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
    loras_applied: Optional[list] = None  # List of LoRAs that were applied


class ImageGenerationService:
    """
    Generates character portraits and scene images using local ComfyUI.
    Images are stored on local filesystem with metadata in database.
    """

    # ══════════════════════════════════════════════════════════════════════════════════════════
    # PONY DIFFUSION V6 XL CHEATSHEET - Based on official guides and community best practices
    # Sources: https://civitai.com/articles/5473/pony-cheatsheet-v2
    #          https://huggingface.co/LyliaEngine/Pony_Diffusion_V6_XL
    # ══════════════════════════════════════════════════════════════════════════════════════════

    # Base model - Illustrious Animilf for consistent mature anime style
    DEFAULT_MODEL = "illustriousAnimilf_v01.safetensors"
    # Alternatives:
    #   "hardcoreHentai_ponyV11.safetensors" - Pony-based NSFW
    #   "ntrMIXIllustriousXL_xiii.safetensors" - Illustrious mix

    # Standard LoRAs for consistent style (chained)
    STANDARD_LORAS = [
        {
            "name": "Youkoso_Sukebe_Elf_No_Mori_E.safetensors",
            "weight": 0.8,
            "trigger": "youkoso_elfstyle"
        },
        {
            "name": "BreastSizeSlider_IL.safetensors",
            "weight": 2.0,  # 2.0 = massive breasts (standard)
            "trigger": None  # Slider - weight controls size (-0.5 to 2.0+)
        }
    ]

    # Optional Pose LoRAs (add to custom_loras when needed)
    POSE_LORAS = {
        "cowgirl": {
            "name": "PSCowgirl.safetensors",
            "weight": 0.9,
            # Trigger: "1boy, penis, squatting cowgirl position, vaginal, pov"
            # - Remove "penis" for deeper penetration effect
            # - Remove "penis, vaginal" for sitting/implied sex effect
            # - Add "squatting" for wider shot showing more position
            "trigger": "1boy, penis, squatting cowgirl position, vaginal, pov"
        }
    }

    # Optional Concept/Outfit LoRAs
    CONCEPT_LORAS = {
        "slutty_clothes": {
            "name": "Concept_Slutty_Clothes.safetensors",
            "weight": 1.0,
            # Trigger: "slutty_clothes" - generates customizable improper outfits
            # Good combos: micro bikini, string bikini, see-through, etc.
            "trigger": "slutty_clothes"
        }
    }

    # Body Shape LoRAs (for Pony models - maintains slim body with large breasts)
    BODY_LORAS = {
        "slim_body": {
            "name": "LargeBreastsSlimBody.safetensors",  # Need to download
            "weight": 1.0,
            # Use with: breasts, large breasts, huge breasts, gigantic breasts
            # Combine with: slim, slender, skinny, narrow waist, petite
            # Recommended for Pony models with AutismMix_confetti
            "trigger": None  # No trigger, just add body shape tags
        },
        "venus_body": {
            "name": "PulenKompot-Venus_Body_IL.safetensors",
            "weight": 1.0,
            # For Illustrious models - exaggerated curves
            "trigger": "pk_venusbody"
        }
    }

    # Legacy single LoRA reference
    STANDARD_LORA = STANDARD_LORAS[0]

    # ═══════════════════════════════════════════════════════════════════════════
    # PONY V6 XL SCORE SYSTEM (REQUIRED - always start prompts with these)
    # ═══════════════════════════════════════════════════════════════════════════
    # score_9         = Highest quality tier
    # score_8_up      = Quality 8 and above
    # score_7_up      = Quality 7 and above
    # score_6_up      = Quality 6 and above (often in negative)
    # score_5_up      = Quality 5 and above (often in negative)
    # score_4_up      = Quality 4 and above (often in negative)
    PONY_QUALITY_POSITIVE = "score_9, score_8_up, score_7_up"
    PONY_QUALITY_NEGATIVE = "score_6, score_5, score_4"

    # ═══════════════════════════════════════════════════════════════════════════
    # PONY V6 XL RATING TAGS (Content control)
    # ═══════════════════════════════════════════════════════════════════════════
    # rating_safe         = SFW content
    # rating_questionable = Borderline/suggestive content
    # rating_explicit     = NSFW/adult content
    RATING_SAFE = "rating_safe"
    RATING_QUESTIONABLE = "rating_questionable"
    RATING_EXPLICIT = "rating_explicit"

    # ═══════════════════════════════════════════════════════════════════════════
    # PONY V6 XL SOURCE TAGS (Style control)
    # ═══════════════════════════════════════════════════════════════════════════
    # source_anime    = Anime aesthetic
    # source_cartoon  = Western cartoon style
    # source_pony     = MLP pony style
    # source_furry    = Furry art style
    SOURCE_ANIME = "source_anime"
    SOURCE_CARTOON = "source_cartoon"
    SOURCE_PONY = "source_pony"
    SOURCE_FURRY = "source_furry"

    # ═══════════════════════════════════════════════════════════════════════════
    # ILLUSTRIOUS XL SETTINGS (NEW STANDARD)
    # ═══════════════════════════════════════════════════════════════════════════
    # Model: illustriousAnimilf_v01.safetensors
    # LoRA: Youkoso_Sukebe_Elf_No_Mori_E.safetensors @ 0.8 (trigger: youkoso_elfstyle)
    # Sampler: dpmpp_2m
    # Scheduler: karras
    # Steps: 30
    # CFG: 7.0
    # HiRes: 25 steps, 0.55 denoise, 1.5x scale
    # Quality tags: masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest

    ILLUSTRIOUS_QUALITY_POSITIVE = "masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest"
    ILLUSTRIOUS_QUALITY_NEGATIVE = "lowres, worst quality, bad quality, bad anatomy, blurry, censored"

    # ═══════════════════════════════════════════════════════════════════════════
    # PONY SETTINGS (LEGACY - for Pony-based models)
    # ═══════════════════════════════════════════════════════════════════════════
    # Sampler: Euler a or DPM++ 2M SDE Karras
    # Steps: 25-35
    # CFG: 5-6
    # Clip Skip: 2 (CRITICAL - or -2 in some software)
    # Resolution: 1024x1024 or other SDXL resolutions

    # LoRAs to enhance generation
    LORAS = {
        # Zheng v5 - QUALITY BOOST LoRA for Pony Diffusion V6 (405MB - highly detailed)
        # Weight 1.0 with "zheng" trigger word in prompt (proven best approach)
        "zheng": {"name": "Zheng v5 -  PonyDiffusion v6.safetensors", "weight": 1.0},
        # Milfication LoRA - triggers with "milfication" tag
        "milfication": {"name": "milfication_pdxl_goofy.safetensors", "weight": 1.0},
        # MatureFemalePony - enhances mature female features
        "mature_female": {"name": "MatureFemalePony.safetensors", "weight": 0.8},
        # Alternative milf LoRAs
        "milf": {"name": "Milf.safetensors", "weight": 0.8},
        "nsfw": {"name": "ponyxl_nsfw.safetensors", "weight": 1.0},
        "detail": {"name": "add_detail_XL.safetensors", "weight": 0.5},
        # Expressive_H - enhanced expressions/emotions
        "expressive": {"name": "Expressive_H-000001.safetensors", "weight": 0.7},
        # Extreme bukkake - specific effect LoRA
        "bukkake": {"name": "extreme_bukkake_v0.1-pony.safetensors", "weight": 0.8},
        # Sagging - body realism enhancement
        "sagging": {"name": "sagging-vpred-v0.7.safetensors", "weight": 0.6},
    }

    # Textual Inversion Embeddings (use in negative prompt as "ng_deepnegative_v1_75t")
    EMBEDDINGS = {
        "deepnegative": "ng_deepnegative_v1_75t",  # Use in negative prompt to improve quality
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # NEGATIVE PROMPTS (Minimal approach recommended by Pony docs)
    # ═══════════════════════════════════════════════════════════════════════════
    # Pony V6 XL is designed to NOT need negative prompts in most cases
    # For NSFW: add censorship removal tags
    # For quality: low scores in negative

    # Minimal negative (recommended by official docs)
    MINIMAL_NEGATIVE = "score_4, score_5, score_6"

    # Extended negative for anti-censorship
    NSFW_NEGATIVE = (
        "score_4, score_5, score_6, "
        "censored, bar censor, mosaic, mosaic censoring, "
        "source_pony, source_furry, source_cartoon"
    )

    # Full safety negative (for mature human content)
    MASTER_NEGATIVE = (
        "score_4, score_5, score_6, "
        "censored, bar censor, mosaic, mosaic censoring, white bar, black bar, "
        "signature, watermark, text, "
        "source_pony, source_furry, source_cartoon, 3d, "
        "(loli:2.5), (child:2.5), (young:2.5), (teen:2.5), "
        "petite, flat chest, small breasts, skinny"
    )

    # Alias for backward compatibility
    MASTER_NEGATIVE_PROMPT = MASTER_NEGATIVE

    # ══════════════════════════════════════════════════════════════════════════
    # STORY-BASED STYLE PRESETS - Pony-optimized with proper tag structure
    # Template: score_tags, source_tag, BREAK, character_description, scene_tags
    # Uses milfication LoRA trigger word for mature female content
    # ══════════════════════════════════════════════════════════════════════════
    STORY_STYLE_PRESETS = {
        # Default high-quality anime style
        "default": {
            "base_positive": f"{PONY_QUALITY_POSITIVE}, {SOURCE_ANIME}",
            "base_negative": MINIMAL_NEGATIVE,
        },

        # Mature female isekai characters - uses milfication LoRA (NSFW DEFAULT)
        "isekai_milf": {
            "base_positive": (
                f"{PONY_QUALITY_POSITIVE}, {RATING_EXPLICIT}, "
                "milfication, milf, wife, mature female, "
                "gigantic breasts, voluptuous body, wide hips, thick thighs, "
                "completely nude, nude, nipples, puffy nipples, large areolae, "
                "looking at viewer, beautiful detailed eyes, expressive face"
            ),
            "base_negative": (
                "realistic, monochrome, greyscale, artist name, signature, watermark, "
                "censored, bar censor, mosaic, mosaic censoring, "
                "source_pony, source_furry, source_cartoon, "
                "(loli:2.5), (child:2.5), (young:2.5), (teen:2.5), "
                "petite, flat chest, small breasts"
            ),
        },

        # Explicit isekai with milfication (full NSFW)
        "isekai_explicit": {
            "base_positive": (
                f"{PONY_QUALITY_POSITIVE}, milfication, milf, wife, "
                "gigantic breasts, female pubic hair, jewelry, navel, "
                "puffy nipples, nude, mature female, "
                "large breasts, dark nipples, 1girl, "
                "huge nipples, large areolae, on bed, "
                "blush, smile, long hair"
            ),
            "base_negative": "realistic, monochrome, greyscale, artist name, signature, watermark",
        },

        # Maximum explicit with milfication - spread pose
        "milfication_spread": {
            "base_positive": (
                f"{PONY_QUALITY_POSITIVE}, milfication, milf, wife, "
                "gigantic breasts, female pubic hair, bed sheet, jewelry, navel, "
                "puffy nipples, heart, spread legs, blush, "
                "excessive pubic hair, nude, mature female, folded, "
                "ass, long hair, lying, legs up, on back, smile, "
                "pussy, anal hair, large breasts, dark nipples, 1girl, "
                "puckered anus, presenting, huge nipples, large areolae, on bed"
            ),
            "base_negative": "realistic, monochrome, greyscale, artist name, signature, watermark",
        },

        # Grimdark warrior (dark fantasy)
        "grimdark_warrior": {
            "base_positive": (
                f"{PONY_QUALITY_POSITIVE}, {SOURCE_ANIME}, BREAK "
                "1girl, solo, mature female warrior, battle-scarred, "
                "muscular yet curvy, detailed armor, grimdark fantasy, "
                "cinematic lighting, atmospheric, dramatic composition"
            ),
            "base_negative": (
                f"{PONY_QUALITY_NEGATIVE}, cute, kawaii, "
                "(loli:2.5), (young:2.5), (child:2.5)"
            ),
        },

        # Wholesome romance (SFW)
        "wholesome_romance": {
            "base_positive": (
                f"{PONY_QUALITY_POSITIVE}, {SOURCE_ANIME}, {RATING_SAFE}, BREAK "
                "1girl, solo, beautiful mature woman, elegant dress, "
                "soft lighting, romantic atmosphere, wholesome, "
                "detailed eyes, gentle expression, warm smile"
            ),
            "base_negative": (
                f"{PONY_QUALITY_NEGATIVE}, nsfw, explicit, nude, aroused"
            ),
        },

        # Dark fantasy erotic with milfication
        "dark_fantasy_erotic": {
            "base_positive": (
                f"{PONY_QUALITY_POSITIVE}, milfication, milf, "
                "mature adult woman, voluptuous body, "
                "gigantic breasts, nude, sensual pose, "
                "dark fantasy setting, dramatic lighting, "
                "detailed skin texture, expressive eyes, sweaty"
            ),
            "base_negative": "realistic, monochrome, greyscale, artist name, signature, watermark",
        },

        # Maximum explicit with all tags
        "maximum_explicit": {
            "base_positive": (
                f"{PONY_QUALITY_POSITIVE}, milfication, milf, wife, "
                "gigantic breasts, female pubic hair, nude, "
                "mature female, spread legs, presenting, "
                "pussy, large breasts, dark nipples, "
                "huge nipples, large areolae, "
                "aroused expression, wet skin, erect nipples, "
                "detailed background, masterpiece"
            ),
            "base_negative": "realistic, monochrome, greyscale, artist name, signature, watermark",
        },
    }

    # ══════════════════════════════════════════════════════════════════════════
    # ART STYLE PRESETS - Combined with story style, ALL USE PONY V6 XL
    # OPTIMIZED based on community best practices (civitai.com)
    # Key: CFG 5, Euler a, SGM Uniform scheduler, CLIP Skip 2, zheng trigger word
    # ══════════════════════════════════════════════════════════════════════════

    # Quality tags - proven combination for best results
    QUALITY_POSITIVE = "masterpiece, best quality, absurdres, highres, very awa, score_9_up, score_8"

    # Minimal negative (proven most effective)
    QUALITY_NEGATIVE = "worst aesthetic, worst quality, low quality, bad quality, lowres, bar censor, censored"

    STYLE_PRESETS = {
        "anime": {
            "quality_tags": ["zheng", SOURCE_ANIME, "anime coloring", "vibrant colors", "detailed eyes"],
            "negative": QUALITY_NEGATIVE,
            "model": DEFAULT_MODEL,
            "sampler": "euler_ancestral",  # Euler a - best for natural results
            "scheduler": "sgm_uniform",    # SGM Uniform scheduler
            "steps": 30,
            "cfg": 5.0,  # CFG 5 - optimal for Pony models
        },
        "realistic": {
            "quality_tags": ["zheng", "photorealistic", "8k uhd", "dslr quality", "detailed skin texture"],
            "negative": QUALITY_NEGATIVE,
            "model": DEFAULT_MODEL,
            "sampler": "euler_ancestral",
            "scheduler": "sgm_uniform",
            "steps": 30,
            "cfg": 5.0,
        },
        "fantasy": {
            "quality_tags": ["zheng", SOURCE_ANIME, "fantasy art", "digital painting", "dramatic lighting"],
            "negative": QUALITY_NEGATIVE,
            "model": DEFAULT_MODEL,
            "sampler": "euler_ancestral",
            "scheduler": "sgm_uniform",
            "steps": 30,
            "cfg": 5.0,
        },
        # Explicit style optimized for NSFW - best quality settings
        "explicit": {
            "quality_tags": [
                "zheng", SOURCE_ANIME,
                "detailed genitalia", "anatomically correct",
                "detailed skin", "detailed nipples", "wet skin"
            ],
            "negative": QUALITY_NEGATIVE,
            "model": DEFAULT_MODEL,
            "sampler": "euler_ancestral",
            "scheduler": "sgm_uniform",
            "steps": 30,
            "cfg": 5.0,
        },
    }

    @classmethod
    def get_story_style(cls, story_id: str = None, story_title: str = None) -> dict:
        """
        Get the appropriate story style preset based on story ID or title.

        Args:
            story_id: Story UUID as string
            story_title: Story title for keyword matching

        Returns:
            Story style preset dict with base_positive and base_negative
        """
        search_text = f"{story_id or ''} {story_title or ''}".lower()

        # Match keywords to story styles
        if any(kw in search_text for kw in ["isekai", "mira", "eros", "mana", "test", "aurelia"]):
            return cls.STORY_STYLE_PRESETS["isekai_milf"]
        elif any(kw in search_text for kw in ["grimdark", "warrior", "battle", "war"]):
            return cls.STORY_STYLE_PRESETS["grimdark_warrior"]
        elif any(kw in search_text for kw in ["romance", "wholesome", "love", "sweet"]):
            return cls.STORY_STYLE_PRESETS["wholesome_romance"]
        elif any(kw in search_text for kw in ["dark", "erotic", "sensual", "adult"]):
            return cls.STORY_STYLE_PRESETS["dark_fantasy_erotic"]

        return cls.STORY_STYLE_PRESETS["default"]

    # Entity type to visual descriptors - NSFW uses full body by default
    ENTITY_TYPE_PROMPTS = {
        EntityType.CHARACTER: "1girl, solo, full body, standing",
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
        use_cloud_storage: bool = True,
    ):
        """
        Initialize image generation service.

        Args:
            comfyui_url: ComfyUI API URL (default from env)
            output_base_path: Base path for generated images (default from env)
            serve_url_base: Base URL for serving images (optional)
            use_cloud_storage: Whether to upload images to B2 cloud storage (default True)
        """
        self.comfyui_url = comfyui_url or os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
        self.output_base = Path(output_base_path or os.getenv("IMAGE_OUTPUT_PATH", "./generated_images"))
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.serve_url_base = serve_url_base
        self.use_cloud_storage = use_cloud_storage

        # Initialize B2 storage if enabled
        self._b2_storage: Optional[B2StorageService] = None
        if use_cloud_storage:
            try:
                self._b2_storage = get_b2_storage()
                print("[ImageGen] B2 cloud storage initialized")
            except Exception as e:
                print(f"[ImageGen] B2 storage unavailable, using local only: {e}")
                self._b2_storage = None

    def build_character_prompt(
        self,
        entity: Entity,
        style: str = "anime",
        additional_tags: list[str] = None,
        pose: str = None,
        expression: str = None,
        story_id: str = None,
        story_title: str = None,
    ) -> tuple[str, str]:
        """
        Build a generation prompt from entity data.

        Args:
            entity: Entity to generate image for
            style: Art style preset
            additional_tags: Extra tags to include
            pose: Specific pose (e.g., "sitting", "action pose")
            expression: Facial expression (e.g., "smiling", "serious")
            story_id: Story UUID for style matching
            story_title: Story title for style matching

        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """
        preset = self.STYLE_PRESETS.get(style, self.STYLE_PRESETS["anime"])

        # Get story-specific style (adult content, theme, etc.)
        story_style = self.get_story_style(story_id, story_title)

        # Start with optimized quality tags (masterpiece, best quality, absurdres, etc.)
        prompt_parts = [self.QUALITY_POSITIVE]

        # Add story base positive (contains adult/theme tags)
        prompt_parts.append(story_style["base_positive"])

        # Add art style quality tags (includes "zheng" trigger word for LoRA)
        prompt_parts.extend(preset["quality_tags"])

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
                                     "slim", "scar", "tattoo", "beard", "young", "old",
                                     "breast", "curvy", "voluptuous", "body", "hip"]
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
            desc_words = entity.description[:200]  # Increased for more detail
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

        # Combine story negative with style negative
        negative = f"{story_style['base_negative']}, {preset['negative']}"

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
        width: int = 832,
        height: int = 1216,
        seed: int = None,
        additional_tags: list[str] = None,
        pose: str = None,
        expression: str = None,
        # Extended HiRes and LoRA parameters
        custom_loras: list[dict] = None,
        use_standard_lora: bool = None,
        use_default_loras: bool = True,
        steps: int = None,
        cfg: float = None,
        hires_scale: float = 1.5,
        hires_denoise: float = 0.55,
    ) -> GenerationResult:
        """
        Generate a character portrait with HiRes upscaling.

        Args:
            entity: Entity to generate portrait for
            entity_id: Entity UUID
            story_id: Story UUID
            style: Art style preset
            width: Base image width (default 832 for portrait)
            height: Base image height (default 1216 for portrait)
            seed: Random seed (None for consistent seed based on name)
            additional_tags: Extra prompt tags
            pose: Specific pose
            expression: Facial expression
            custom_loras: List of custom LoRAs [{'name': 'file.safetensors', 'weight': 0.7}]
            use_standard_lora: Whether to use standard anime LoRA
            use_default_loras: Whether to use auto-detected default LoRAs
            steps: Generation steps (default from preset)
            cfg: CFG scale (default from preset)
            hires_scale: HiRes upscale factor (default 1.5)
            hires_denoise: HiRes denoising strength (default 0.55)

        Returns:
            GenerationResult with file path and metadata
        """
        preset = self.STYLE_PRESETS.get(style, self.STYLE_PRESETS["anime"])

        # Build prompts (with story-specific style)
        positive, negative = self.build_character_prompt(
            entity=entity,
            style=style,
            additional_tags=additional_tags,
            pose=pose,
            expression=expression,
            story_id=str(story_id),
        )

        # Use consistent seed if not provided
        if seed is None:
            seed = self._get_seed_for_entity(entity.canonical_name)

        # Use preset values if not overridden
        actual_steps = steps if steps is not None else preset["steps"]
        actual_cfg = cfg if cfg is not None else preset["cfg"]

        # Determine whether to use default LoRAs
        use_loras = use_default_loras if use_default_loras is not None else True

        # Build HiRes workflow for quality output
        workflow = self._build_hires_workflow(
            positive=positive,
            negative=negative,
            width=width,
            height=height,
            seed=seed,
            model=preset["model"],
            sampler=preset["sampler"],
            steps=actual_steps,
            cfg=actual_cfg,
            scheduler=preset.get("scheduler", "sgm_uniform"),
            use_loras=use_loras,
            hires_scale=hires_scale,
            hires_steps=int(actual_steps * 0.4),  # 40% of base steps for HiRes
            hires_denoise=hires_denoise,
            custom_loras=custom_loras,
        )

        # Execute generation (HiRes takes longer)
        image_data = await self._execute_workflow(workflow, timeout=600)

        # Calculate final dimensions
        final_width = int(width * hires_scale)
        final_height = int(height * hires_scale)

        # Save to filesystem
        output_dir = self.output_base / str(story_id) / "entities" / str(entity_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portrait_{style}_{seed}_{timestamp}.png"
        file_path = output_dir / filename

        with open(file_path, "wb") as f:
            f.write(image_data)

        # Upload to B2 cloud storage if enabled
        file_url = None
        cloud_path = None

        if self._b2_storage:
            try:
                b2_result = self._b2_storage.upload_image(
                    file_data=image_data,
                    story_id=str(story_id),
                    entity_id=str(entity_id),
                    image_type="portrait",
                    style=style,
                    extension="png"
                )
                file_url = b2_result["file_url"]  # Signed URL
                cloud_path = b2_result["file_path"]
                print(f"[ImageGen] Portrait uploaded to B2: {cloud_path}")
            except Exception as e:
                print(f"[ImageGen] B2 upload failed, using local path: {e}")

        # Fallback to local serve URL if B2 upload failed or not enabled
        if file_url is None and self.serve_url_base:
            relative_path = file_path.relative_to(self.output_base)
            file_url = f"{self.serve_url_base}/{relative_path}"

        # Build loras_applied list for response
        loras_applied = custom_loras if custom_loras else []

        return GenerationResult(
            file_path=cloud_path or str(file_path),  # Prefer cloud path if available
            file_url=file_url,
            generation_prompt=positive,
            negative_prompt=negative,
            seed=seed,
            model_used=preset["model"],
            sampler=preset["sampler"],
            steps=actual_steps,
            cfg_scale=actual_cfg,
            width=final_width,
            height=final_height,
            loras_applied=loras_applied,
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

        # Build and execute workflow with research-optimized settings
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
            scheduler=preset.get("scheduler", "karras"),  # Use Karras scheduler for quality
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

        # Upload to B2 cloud storage if enabled
        file_url = None
        cloud_path = None

        if self._b2_storage:
            try:
                b2_result = self._b2_storage.upload_scene_image(
                    file_data=image_data,
                    story_id=str(story_id),
                    session_id=str(session_id) if session_id else None,
                    scene_description=description[:256],
                    extension="png"
                )
                file_url = b2_result["file_url"]  # Signed URL
                cloud_path = b2_result["file_path"]
                print(f"[ImageGen] Scene uploaded to B2: {cloud_path}")
            except Exception as e:
                print(f"[ImageGen] B2 upload failed, using local path: {e}")

        # Fallback to local serve URL if B2 upload failed or not enabled
        if file_url is None and self.serve_url_base:
            relative_path = file_path.relative_to(self.output_base)
            file_url = f"{self.serve_url_base}/{relative_path}"

        return GenerationResult(
            file_path=cloud_path or str(file_path),  # Prefer cloud path if available
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
        use_loras: bool = True,
        scheduler: str = "karras",  # Karras scheduler for better quality
    ) -> dict:
        """Build ComfyUI txt2img workflow JSON with LoRA support."""

        # Base workflow without LoRAs (still uses CLIP Skip 2 for Pony quality)
        if not use_loras:
            return {
                "3": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": seed,
                        "steps": steps,
                        "cfg": cfg,
                        "sampler_name": sampler,
                        "scheduler": scheduler,  # Karras scheduler for better quality
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
                # CLIP Skip 2 - CRITICAL for Pony Diffusion V6 XL quality
                "14": {
                    "class_type": "CLIPSetLastLayer",
                    "inputs": {
                        "stop_at_clip_layer": -2,  # CLIP Skip 2
                        "clip": ["4", 1]
                    }
                },
                "6": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "text": positive,
                        "clip": ["14", 0]  # Connect to CLIP Skip output
                    }
                },
                "7": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "text": negative,
                        "clip": ["14", 0]  # Connect to CLIP Skip output
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

        # Workflow with LoRAs + CLIP Skip 2:
        # Checkpoint (4) -> Zheng (16) -> MatureFemale (10) -> NSFW (11) -> Milfication (12) -> Detail (13) -> CLIP Skip (15)
        zheng_lora = self.LORAS["zheng"]
        mature_female_lora = self.LORAS["mature_female"]
        nsfw_lora = self.LORAS["nsfw"]
        milfication_lora = self.LORAS["milfication"]
        detail_lora = self.LORAS["detail"]

        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler,
                    "scheduler": scheduler,  # Karras scheduler for better quality
                    "denoise": 1.0,
                    "model": ["13", 0],  # Connect to last LoRA (Detail) output
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
            # CLIP Skip 2 - CRITICAL for Pony Diffusion V6 XL quality
            # Applied after LoRA chain to preserve LoRA effects
            "15": {
                "class_type": "CLIPSetLastLayer",
                "inputs": {
                    "stop_at_clip_layer": -2,  # CLIP Skip 2
                    "clip": ["13", 1]  # Connect to last LoRA CLIP output
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": positive,
                    "clip": ["15", 0]  # Connect to CLIP Skip output
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative,
                    "clip": ["15", 0]  # Connect to CLIP Skip output
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
            },
            # LoRA 1: Zheng v5 - QUALITY BOOST (first in chain for best quality impact)
            "16": {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": zheng_lora["name"],
                    "strength_model": zheng_lora["weight"],
                    "strength_clip": zheng_lora["weight"],
                    "model": ["4", 0],
                    "clip": ["4", 1]
                }
            },
            # LoRA 2: MatureFemalePony - base mature female enhancement
            "10": {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": mature_female_lora["name"],
                    "strength_model": mature_female_lora["weight"],
                    "strength_clip": mature_female_lora["weight"],
                    "model": ["16", 0],  # Connect to Zheng output
                    "clip": ["16", 1]
                }
            },
            # LoRA 3: NSFW LoRA - enhances explicit content generation
            "11": {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": nsfw_lora["name"],
                    "strength_model": nsfw_lora["weight"],
                    "strength_clip": nsfw_lora["weight"],
                    "model": ["10", 0],
                    "clip": ["10", 1]
                }
            },
            # LoRA 4: Milfication LoRA (trigger word: "milfication")
            "12": {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": milfication_lora["name"],
                    "strength_model": milfication_lora["weight"],
                    "strength_clip": milfication_lora["weight"],
                    "model": ["11", 0],
                    "clip": ["11", 1]
                }
            },
            # LoRA 5: Detail LoRA (last in chain for final detail enhancement)
            "13": {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": detail_lora["name"],
                    "strength_model": detail_lora["weight"],
                    "strength_clip": detail_lora["weight"],
                    "model": ["12", 0],
                    "clip": ["12", 1]
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

    def _build_hires_workflow(
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
        scheduler: str = "sgm_uniform",
        use_loras: bool = True,
        hires_scale: float = 1.5,
        hires_steps: int = 10,
        hires_denoise: float = 0.3,
        upscaler_model: str = "4x-UltraSharp.pth",
        custom_loras: list[dict] = None,
        vae_model: str = None,  # Optional separate VAE (e.g., "sdxl_vae_fp16_fix.safetensors")
        use_face_detailer: bool = False,  # Enable FaceDetailer for sharper faces
    ) -> dict:
        """
        Build ComfyUI workflow with HiRes upscaling.

        This mimics the A1111 HiRes Fix workflow:
        1. Generate base image at specified resolution
        2. Upscale latent with bislerp
        3. Run second KSampler pass with low denoise

        Args:
            positive: Positive prompt
            negative: Negative prompt
            width: Base width
            height: Base height
            seed: Random seed
            model: Checkpoint model name
            sampler: Sampler name
            steps: Base generation steps
            cfg: CFG scale
            scheduler: Scheduler name
            use_loras: Whether to use default LoRA chain
            hires_scale: Upscale factor (default 1.5x)
            hires_steps: Steps for HiRes pass
            hires_denoise: Denoise strength for HiRes pass (0.3 = subtle refinement)
            upscaler_model: ESRGAN upscaler model name (unused with latent upscale)
            custom_loras: List of custom LoRAs [{'name': 'file.safetensors', 'weight': 1.0}]
        """
        # Calculate target HiRes dimensions
        target_width = int(width * hires_scale)
        target_height = int(height * hires_scale)

        # Determine if we're using any LoRAs
        has_custom_loras = custom_loras and len(custom_loras) > 0
        has_any_loras = use_loras or has_custom_loras

        # Get default LoRA settings if using them
        if use_loras:
            zheng_lora = self.LORAS["zheng"]
            mature_female_lora = self.LORAS["mature_female"]
            nsfw_lora = self.LORAS["nsfw"]
            milfication_lora = self.LORAS["milfication"]
            detail_lora = self.LORAS["detail"]

        # Determine which model/clip output to use based on LoRA configuration
        if use_loras:
            # Using default LoRA chain - output from node 13 (last default LoRA)
            model_output = ["13", 0]
            clip_output = ["15", 0]
        elif has_custom_loras:
            # Using custom LoRAs - output from last custom LoRA node
            last_lora_node = str(100 + len(custom_loras) - 1)
            model_output = [last_lora_node, 0]
            clip_output = ["15", 0]  # CLIP skip after custom LoRAs
        else:
            # No LoRAs - direct from checkpoint
            model_output = ["4", 0]
            clip_output = ["14", 0]

        # Determine VAE source - use separate VAE if provided, otherwise from checkpoint
        vae_source = ["30", 0] if vae_model else ["4", 2]

        workflow = {
            # ======= BASE GENERATION =======
            # Checkpoint loader
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": model
                }
            },
            # Empty latent for base generation
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            # Base KSampler
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                    "model": model_output,
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                }
            },
            # VAE Decode base image
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": vae_source
                }
            },

            # ======= HIRES UPSCALING =======
            # Upscale latent directly (simpler and more reliable than ESRGAN pipeline)
            "22": {
                "class_type": "LatentUpscale",
                "inputs": {
                    "samples": ["3", 0],
                    "upscale_method": "bislerp",
                    "width": target_width,
                    "height": target_height,
                    "crop": "disabled"
                }
            },
            # HiRes KSampler (refinement pass)
            "24": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": hires_steps,
                    "cfg": cfg,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": hires_denoise,
                    "model": model_output,
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["22", 0]  # Connect to LatentUpscale output
                }
            },
            # Final VAE Decode
            "25": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["24", 0],
                    "vae": vae_source
                }
            },
            # Save final image
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "ComfyUI_HiRes",
                    "images": ["25", 0]
                }
            },
        }

        # Add CLIP Skip node - connect to appropriate source
        if use_loras:
            workflow["15"] = {
                "class_type": "CLIPSetLastLayer",
                "inputs": {
                    "stop_at_clip_layer": -2,
                    "clip": ["13", 1]  # After default LoRA chain
                }
            }
        elif has_custom_loras:
            last_lora_node = str(100 + len(custom_loras) - 1)
            workflow["15"] = {
                "class_type": "CLIPSetLastLayer",
                "inputs": {
                    "stop_at_clip_layer": -2,
                    "clip": [last_lora_node, 1]  # After custom LoRA chain
                }
            }
        else:
            workflow["14"] = {
                "class_type": "CLIPSetLastLayer",
                "inputs": {
                    "stop_at_clip_layer": -2,
                    "clip": ["4", 1]  # Direct from checkpoint
                }
            }

        # Add text encoders
        workflow["6"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": positive,
                "clip": clip_output
            }
        }
        workflow["7"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative,
                "clip": clip_output
            }
        }

        # Add default LoRA chain if enabled
        if use_loras:
            workflow["16"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": zheng_lora["name"],
                    "strength_model": zheng_lora["weight"],
                    "strength_clip": zheng_lora["weight"],
                    "model": ["4", 0],
                    "clip": ["4", 1]
                }
            }
            workflow["10"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": mature_female_lora["name"],
                    "strength_model": mature_female_lora["weight"],
                    "strength_clip": mature_female_lora["weight"],
                    "model": ["16", 0],
                    "clip": ["16", 1]
                }
            }
            workflow["11"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": nsfw_lora["name"],
                    "strength_model": nsfw_lora["weight"],
                    "strength_clip": nsfw_lora["weight"],
                    "model": ["10", 0],
                    "clip": ["10", 1]
                }
            }
            workflow["12"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": milfication_lora["name"],
                    "strength_model": milfication_lora["weight"],
                    "strength_clip": milfication_lora["weight"],
                    "model": ["11", 0],
                    "clip": ["11", 1]
                }
            }
            workflow["13"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": detail_lora["name"],
                    "strength_model": detail_lora["weight"],
                    "strength_clip": detail_lora["weight"],
                    "model": ["12", 0],
                    "clip": ["12", 1]
                }
            }

        # Add custom LoRAs if provided (and not using default LoRAs)
        if has_custom_loras and not use_loras:
            prev_model = ["4", 0]
            prev_clip = ["4", 1]
            for i, lora in enumerate(custom_loras):
                node_id = str(100 + i)
                lora_name = lora.get("name", "")
                lora_weight = lora.get("weight", 1.0)
                workflow[node_id] = {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "lora_name": lora_name,
                        "strength_model": lora_weight,
                        "strength_clip": lora_weight,
                        "model": prev_model,
                        "clip": prev_clip
                    }
                }
                prev_model = [node_id, 0]
                prev_clip = [node_id, 1]

        # Add separate VAE loader if specified
        if vae_model:
            workflow["30"] = {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": vae_model
                }
            }

        # Add FaceDetailer if enabled (for sharper faces)
        if use_face_detailer:
            # Update SaveImage to use FaceDetailer output instead
            workflow["9"]["inputs"]["images"] = ["40", 0]

            # UltralyticsDetectorProvider for face detection
            workflow["31"] = {
                "class_type": "UltralyticsDetectorProvider",
                "inputs": {
                    "model_name": "face_yolov8n.pt"
                }
            }

            # SAMLoader for segmentation (optional, enhances mask quality)
            # workflow["32"] = {
            #     "class_type": "SAMLoader",
            #     "inputs": {
            #         "model_name": "sam_vit_b_01ec64.pth",
            #         "device_mode": "AUTO"
            #     }
            # }

            # FaceDetailer node
            workflow["40"] = {
                "class_type": "FaceDetailer",
                "inputs": {
                    "image": ["25", 0],  # Input from HiRes VAE decode
                    "model": model_output,
                    "clip": clip_output,
                    "vae": vae_source,
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "bbox_detector": ["31", 0],
                    "sam_model_opt": None,
                    "segm_detector_opt": None,
                    "detailer_hook": None,
                    "guide_size": 512,
                    "guide_size_for": True,
                    "max_size": 1024,
                    "seed": seed,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "dpmpp_2m_sde",
                    "scheduler": "karras",
                    "denoise": 0.4,
                    "feather": 5,
                    "noise_mask": True,
                    "force_inpaint": True,
                    "bbox_threshold": 0.3,
                    "bbox_dilation": 10,
                    "bbox_crop_factor": 3.0,
                    "sam_detection_hint": "center-1",
                    "sam_dilation": 0,
                    "sam_threshold": 0.93,
                    "sam_bbox_expansion": 0,
                    "sam_mask_hint_threshold": 0.7,
                    "sam_mask_hint_use_negative": "False",
                    "drop_size": 10,
                    "wildcard": "",
                    "cycle": 1
                }
            }

        return workflow

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

                # Find the SaveImage node output - prefer node "9" (our standard output node)
                # but fall back to the last node with images if not found
                image_info = None

                # First try to get node "9" specifically (our designated output node)
                if "9" in outputs and "images" in outputs["9"]:
                    image_info = outputs["9"]["images"][0]
                else:
                    # Fall back to finding any node with images (get the last one)
                    for node_id, output in sorted(outputs.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0, reverse=True):
                        if "images" in output:
                            image_info = output["images"][0]
                            break

                if image_info:
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
            story_id=str(story_id),
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

        # Upload to B2 cloud storage if enabled
        file_url = None
        cloud_path = None

        if self.image_service._b2_storage:
            try:
                b2_result = self.image_service._b2_storage.upload_image(
                    file_data=image_data,
                    story_id=str(story_id),
                    entity_id=str(entity_id),
                    image_type="portrait_variation",
                    style=style,
                    extension="png"
                )
                file_url = b2_result["file_url"]  # Signed URL
                cloud_path = b2_result["file_path"]
                print(f"[ImageGen] Portrait variation uploaded to B2: {cloud_path}")
            except Exception as e:
                print(f"[ImageGen] B2 upload failed, using local path: {e}")

        # Fallback to local serve URL if B2 upload failed or not enabled
        if file_url is None and self.image_service.serve_url_base:
            relative_path = file_path.relative_to(self.image_service.output_base)
            file_url = f"{self.image_service.serve_url_base}/{relative_path}"

        result = GenerationResult(
            file_path=cloud_path or str(file_path),  # Prefer cloud path if available
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
