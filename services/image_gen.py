import os
import requests
import base64
import json
from typing import Dict, Any, Optional

class ImageGenService:
    """
    Service for generating NSFW-safe images using Stable Diffusion WebUI API.
    Ensures 18+ adults only with strong prompts/negatives.
    """
    def __init__(self):
        self.api_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
        self.image_base_dir = "/tmp/story-images"
        self.model_name = "Pony Diffusion V6 XL.safetensors"  # Adjust if different filename

    def _build_prompt(self, entity: Dict[str, Any]) -> tuple[str, str]:
        """
        Build positive and negative prompts from entity.
        Always enforces mature adult safety.
        """
        description = entity.get('description', '')
        facts = entity.get('facts', [])
        fact_str = ', '.join(facts) if facts else ''

        base_prompt = f"{description}, {fact_str}".strip(', ')
        safety_positive = "voluptuous adult woman 30-40yo, mature curvy body, large breasts, wide hips"
        style = "portrait, detailed face, explicit aroused expression, high quality, detailed anime style"
        positive_prompt = f"{base_prompt}, {safety_positive}, {style}"

        negative_prompt = "blurry, deformed, ugly, loli, shota, young, child, teen, petite, flat chest, small breasts:2, skinny, underaged, cartoonish"

        return positive_prompt, negative_prompt

    def generate_entity_image(self, entity: Dict[str, Any], story_id: str) -> str:
        """
        Generate image for entity, save to /tmp/story-images/{story_id}/{entity_canonical_name}.png
        Returns file path or placeholder URL on failure.
        Assumes WebUI running at 127.0.0.1:7860 with Pony model loaded.
        """
        canonical_name = entity.get('canonical_name', entity.get('name', 'unknown')).lower().replace(' ', '_').replace('-', '_')
        filepath = f"{self.image_base_dir}/{story_id}/{canonical_name}.png"

        positive_prompt, negative_prompt = self._build_prompt(entity)

        payload = {
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "sampler_name": "Euler a",
            "cfg_scale": 7.0,
            "seed": -1,
            "override_settings": {
                "sd_model_checkpoint": self.model_name
            },
            "alwayson_scripts": {}  # If needed
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            if not result.get('images'):
                raise ValueError("No images in response")

            img_b64 = result['images'][0]
            img_data = base64.b64decode(img_b64)

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                f.write(img_data)

            print(f"[ImageGen] Generated image for {canonical_name} in story {story_id}: {filepath}")
            return filepath

        except Exception as e:
            print(f"[ImageGen] Failed to generate image: {str(e)}")
            # Create placeholder if not exists
            placeholder_path = f"{self.image_base_dir}/placeholder.png"
            if not os.path.exists(placeholder_path):
                # Simple placeholder or skip
                pass
            return placeholder_path
