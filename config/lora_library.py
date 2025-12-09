"""
LoRA Library Configuration

A structured database of available LoRAs with metadata for auto-detection
and dynamic selection based on prompt content.

Each LoRA entry contains:
- filename: The actual .safetensors file name
- display_name: Human-readable name for UI
- category: Grouping for UI organization
- trigger_words: Words that activate this LoRA (for prompt embedding)
- detection_keywords: Keywords in prompt that suggest using this LoRA
- default_weight: Recommended weight (0.0-2.0)
- compatible_models: List of compatible base models
- description: What this LoRA does
- conflicts_with: LoRAs that shouldn't be combined with this one
- auto_enable: If True, automatically enable when keywords detected
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import os


@dataclass
class LoRAEntry:
    """A single LoRA in the library."""
    filename: str
    display_name: str
    category: str
    trigger_words: list[str] = field(default_factory=list)
    detection_keywords: list[str] = field(default_factory=list)
    default_weight: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 2.0
    compatible_models: list[str] = field(default_factory=lambda: ["illustrious", "pony", "sdxl"])
    description: str = ""
    conflicts_with: list[str] = field(default_factory=list)
    auto_enable: bool = False
    nsfw: bool = True

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "display_name": self.display_name,
            "category": self.category,
            "trigger_words": self.trigger_words,
            "detection_keywords": self.detection_keywords,
            "default_weight": self.default_weight,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "compatible_models": self.compatible_models,
            "description": self.description,
            "conflicts_with": self.conflicts_with,
            "auto_enable": self.auto_enable,
            "nsfw": self.nsfw,
        }


# ==============================================
# LORA LIBRARY DATABASE
# ==============================================

LORA_LIBRARY: list[LoRAEntry] = [
    # ============ STYLE / CHARACTER ============
    LoRAEntry(
        filename="Youkoso_Sukebe_Elf_No_Mori_E.safetensors",
        display_name="Youkoso Elf",
        category="style",
        trigger_words=["youkoso"],
        detection_keywords=["elf", "fantasy", "pointy ears"],
        default_weight=0.7,
        description="Youkoso Sukebe Elf style - great for mature fantasy characters",
        auto_enable=False,
    ),
    LoRAEntry(
        filename="Zheng v5 -  PonyDiffusion v6.safetensors",
        display_name="Zheng Style",
        category="style",
        trigger_words=["zheng"],
        detection_keywords=["zheng", "artist:zheng"],
        default_weight=0.8,
        compatible_models=["pony"],
        description="Zheng artist style - distinctive art style",
    ),
    LoRAEntry(
        filename="Expressive_H-000001.safetensors",
        display_name="Expressive H",
        category="style",
        trigger_words=[],
        detection_keywords=["expressive", "emotional"],
        default_weight=0.6,
        description="More expressive faces and emotions",
    ),
    LoRAEntry(
        filename="Leslie_-_Demon_Deals.safetensors",
        display_name="Leslie Demon Deals",
        category="style",
        trigger_words=["leslie"],
        detection_keywords=["demon", "succubus", "horns", "demon girl"],
        default_weight=0.7,
        description="Leslie from Demon Deals - demon/succubus style",
    ),
    LoRAEntry(
        filename="ColoredSkinSuccubus_v10.safetensors",
        display_name="Colored Skin Succubus",
        category="style",
        trigger_words=["colored skin", "succubus"],
        detection_keywords=["colored skin", "purple skin", "blue skin", "red skin"],
        default_weight=0.8,
        description="Succubus with colored/fantasy skin tones - use for exotic skin colors",
        auto_enable=False,  # Only auto-enable for explicit colored skin mentions
        conflicts_with=["Succubus-Illustrious-v01.safetensors"],
    ),
    LoRAEntry(
        filename="Succubus-Illustrious-v01.safetensors",
        display_name="Succubus (Illustrious)",
        category="style",
        trigger_words=["succubus"],
        detection_keywords=["succubus", "demon girl", "wings", "horns", "tail"],
        default_weight=0.7,
        compatible_models=["illustrious"],
        description="Succubus character style for Illustrious models",
        auto_enable=True,
        conflicts_with=["ColoredSkinSuccubus_v10.safetensors"],
    ),
    LoRAEntry(
        filename="MilkFactoryStyleLoRa-01.safetensors",
        display_name="Milk Factory Style",
        category="style",
        trigger_words=["milk factory"],
        detection_keywords=["lactation", "milking", "milk", "breast milk"],
        default_weight=0.7,
        description="Milk factory/lactation art style",
    ),

    # ============ BODY TYPE ============
    LoRAEntry(
        filename="BreastSizeSlider_IL.safetensors",
        display_name="Breast Size Slider",
        category="body",
        trigger_words=["flat chest", "medium breasts", "large breasts", "huge breasts", "gigantic breasts"],
        detection_keywords=["huge breasts", "gigantic breasts", "flat chest", "small breasts"],  # More specific
        default_weight=0.8,
        min_weight=-1.0,  # Negative = smaller
        max_weight=1.5,
        compatible_models=["illustrious"],
        description="Control breast size with weight (-1 to 1.5). Negative = smaller.",
        auto_enable=False,  # Too generic, user should manually select
    ),
    LoRAEntry(
        filename="Milf.safetensors",
        display_name="MILF Body",
        category="body",
        trigger_words=["milf"],
        detection_keywords=["milf"],  # Reduced to avoid false positives on "mature", "mom"
        default_weight=0.7,
        description="Mature female body type",
        auto_enable=True,
        conflicts_with=["MatureFemalePony.safetensors"],
    ),
    LoRAEntry(
        filename="MatureFemalePony.safetensors",
        display_name="Mature Female (Pony)",
        category="body",
        trigger_words=["mature female"],
        detection_keywords=["mature", "older woman", "milf"],
        default_weight=0.6,
        compatible_models=["pony"],
        description="Mature female features for Pony models",
        conflicts_with=["Milf.safetensors"],
    ),
    LoRAEntry(
        filename="PulenKompot-Venus_Body_IL.safetensors",
        display_name="Venus Body",
        category="body",
        trigger_words=["venus body"],
        detection_keywords=["curvy", "thicc", "voluptuous", "plus size"],
        default_weight=0.7,
        compatible_models=["illustrious"],
        description="Curvier, more voluptuous body type",
    ),
    LoRAEntry(
        filename="sagging-vpred-v0.7.safetensors",
        display_name="Sagging Breasts",
        category="body",
        trigger_words=["sagging breasts"],
        detection_keywords=["sagging", "saggy", "natural breasts", "heavy breasts"],
        default_weight=0.5,
        description="More realistic sagging breast physics",
        auto_enable=True,
    ),
    LoRAEntry(
        filename="CLB-PD-v1.safetensors",
        display_name="Clothes Bursting Breasts",
        category="body",
        trigger_words=["bursting breasts", "cleavage", "tight clothes", "skindentation"],
        detection_keywords=["bursting breasts", "tight clothes", "skindentation", "cleavage", "straining buttons", "tight shirt"],
        default_weight=1.0,
        compatible_models=["pony", "illustrious", "sdxl"],
        description="Breasts bursting out of tight clothing effect",
        auto_enable=True,
    ),
    LoRAEntry(
        filename="milfication_pdxl_goofy.safetensors",
        display_name="Milfication",
        category="body",
        trigger_words=["milfication"],
        detection_keywords=["milf", "mature", "older"],
        default_weight=0.5,
        compatible_models=["pony", "sdxl"],
        description="Transform characters to look more mature",
    ),
    LoRAEntry(
        filename="oppai_v0.1-pony.safetensors",
        display_name="Oppai",
        category="body",
        trigger_words=["oppai"],
        detection_keywords=["huge breasts", "massive breasts", "oppai", "big boobs"],
        default_weight=0.6,
        compatible_models=["pony"],
        description="Enhanced large breast rendering for Pony",
    ),

    # ============ CLOTHING ============
    LoRAEntry(
        filename="Concept_Slutty_Clothes.safetensors",
        display_name="Slutty Clothes",
        category="clothing",
        trigger_words=["slutty outfit", "revealing clothes"],
        detection_keywords=["slutty", "revealing", "skimpy", "micro bikini", "string bikini"],
        default_weight=0.6,
        description="More revealing/provocative clothing styles",
        auto_enable=True,
    ),
    LoRAEntry(
        filename="cowprint.safetensors",
        display_name="Cow Print",
        category="clothing",
        trigger_words=["cow print", "cowprint"],
        detection_keywords=["cow print", "cowprint", "cow bikini", "cow girl costume"],
        default_weight=0.8,
        description="Cow print patterns for clothing/bikinis",
    ),
    LoRAEntry(
        filename="EyepatchLeotardV1.safetensors",
        display_name="Eyepatch Leotard",
        category="clothing",
        trigger_words=["eyepatch leotard"],
        detection_keywords=["leotard", "eyepatch", "revealing", "sling bikini"],
        default_weight=0.8,
        description="Revealing eyepatch/sling style leotard",
    ),

    # ============ POSES ============
    LoRAEntry(
        filename="PSCowgirl.safetensors",
        display_name="Cowgirl Position (PS)",
        category="pose",
        trigger_words=["cowgirl position"],
        detection_keywords=["cowgirl", "riding", "girl on top", "straddling"],
        default_weight=1.0,
        description="Cowgirl/riding sex position (smaller LoRA)",
        conflicts_with=["CowGirl.safetensors"],
    ),
    LoRAEntry(
        filename="CowGirl.safetensors",
        display_name="Cowgirl Position",
        category="pose",
        trigger_words=["cowgirl position"],
        detection_keywords=["cowgirl", "riding", "girl on top", "straddling", "reverse cowgirl"],
        default_weight=0.8,
        description="Cowgirl/riding sex position (full LoRA)",
        conflicts_with=["PSCowgirl.safetensors"],
    ),
    LoRAEntry(
        filename="MatingFaceER.safetensors",
        display_name="Mating Press (Face)",
        category="pose",
        trigger_words=["MatingFaceER"],
        detection_keywords=["mating press", "missionary", "legs up"],
        default_weight=1.0,
        description="Mating press position with visible face",
    ),
    LoRAEntry(
        filename="qqq-yuri-cunnilingus-v4.safetensors",
        display_name="Yuri Cunnilingus",
        category="pose",
        trigger_words=["cunnilingus"],
        detection_keywords=["cunnilingus", "oral", "yuri", "lesbian", "face between legs"],
        default_weight=0.8,
        description="Lesbian oral sex position",
    ),
    LoRAEntry(
        filename="V2_IL_Lesbian_Strap-on_sex_in_bed.safetensors",
        display_name="Lesbian Strap-on",
        category="pose",
        trigger_words=["strap-on", "strapon"],
        detection_keywords=["strap-on", "strapon", "lesbian sex", "yuri sex", "dildo"],
        default_weight=0.8,
        compatible_models=["illustrious"],
        description="Lesbian strap-on sex positions",
    ),
    LoRAEntry(
        filename="IL_Lesbian_double_dildo.safetensors",
        display_name="Lesbian Double Dildo",
        category="pose",
        trigger_words=["double dildo"],
        detection_keywords=["double dildo", "shared dildo", "lesbian", "yuri", "tribadism"],
        default_weight=0.8,
        compatible_models=["illustrious"],
        description="Lesbian double-ended dildo positions",
    ),
    LoRAEntry(
        filename="pussysandwich.safetensors",
        display_name="Pussy Sandwich",
        category="pose",
        trigger_words=["pussy sandwich"],
        detection_keywords=["pussy sandwich", "scissoring", "tribbing", "tribadism", "lesbian", "yuri"],
        default_weight=0.8,
        description="Lesbian tribbing/scissoring positions",
    ),
    LoRAEntry(
        filename="PovGroupSex_v10.safetensors",
        display_name="POV Group Sex",
        category="pose",
        trigger_words=["pov group sex"],
        detection_keywords=["pov", "group sex", "gangbang", "multiple boys", "orgy"],
        default_weight=0.8,
        description="POV perspective for group sex scenes",
    ),
    LoRAEntry(
        filename="MultipleGirlsGroup.safetensors",
        display_name="Multiple Girls Group",
        category="pose",
        trigger_words=["multiple girls"],
        detection_keywords=["2girls", "3girls", "multiple girls", "group", "yuri", "lesbian"],
        default_weight=0.7,
        description="Better composition for multiple girls in scene",
        auto_enable=True,
    ),

    # ============ EFFECTS ============
    LoRAEntry(
        filename="extreme_bukkake_v0.1-pony.safetensors",
        display_name="Bukkake Effect",
        category="effect",
        trigger_words=["bukkake", "cum"],
        detection_keywords=["bukkake", "cum", "facial", "covered in cum"],
        default_weight=0.7,
        compatible_models=["pony"],
        description="Bukkake/cum effects",
    ),
    LoRAEntry(
        filename="ponyxl_nsfw.safetensors",
        display_name="Pony NSFW Enhance",
        category="effect",
        trigger_words=[],
        detection_keywords=["explicit", "sex", "penetration"],
        default_weight=0.5,
        compatible_models=["pony"],
        description="Enhanced NSFW details for Pony models",
    ),

    # ============ QUALITY ============
    LoRAEntry(
        filename="add_detail_XL.safetensors",
        display_name="Add Detail XL",
        category="quality",
        trigger_words=[],
        detection_keywords=["detailed", "high detail", "intricate"],
        default_weight=0.5,
        description="Adds more fine details to the image",
    ),

    # ============ NEW - EXPRESSION / POSE ============
    LoRAEntry(
        filename="Ahegaoo.safetensors",
        display_name="Ahegao",
        category="effect",
        trigger_words=["ahegao", "rolling eyes", "blush", "tongue out", "drooling", "saliva"],
        detection_keywords=["ahegao", "tongue out", "rolling eyes", "fucked silly", "cross-eyed", "drooling"],
        default_weight=0.8,
        description="Ahegao facial expression - rolling eyes, tongue out, drooling",
    ),
    LoRAEntry(
        filename="BSQ_v1.safetensors",
        display_name="Breast Squeeze",
        category="pose",
        trigger_words=["breasts squeezed together"],
        detection_keywords=["breast squeeze", "breasts together", "paizuri", "titfuck", "breasts squeezed"],
        default_weight=0.8,
        compatible_models=["illustrious"],
        description="Breast squeeze pose - breasts pressed/squeezed together",
    ),
    LoRAEntry(
        filename="masturbation_female.safetensors",
        display_name="Female Masturbation (Fingering)",
        category="pose",
        trigger_words=["fingering", "schlick", "masturbation"],
        detection_keywords=["masturbation", "masturbating", "fingering", "touching self", "solo female", "schlick"],
        default_weight=0.8,
        description="Female masturbation - fingering poses",
    ),
    LoRAEntry(
        filename="masturbation-v1.safetensors",
        display_name="Female Masturbation (Boob & Fingering)",
        category="pose",
        trigger_words=["masturbation", "fingering", "female_masturbation", "grabbing_own_breast"],
        detection_keywords=["masturbation", "masturbating", "self pleasure", "grabbing breast", "fondling"],
        default_weight=0.8,
        description="Female masturbation - boob fondling and fingering",
    ),
]


class LoRALibrary:
    """Manager for the LoRA library."""

    def __init__(self, lora_directory: str = None):
        self.lora_directory = lora_directory or os.path.expanduser("~/Projects/ComfyUI/models/loras")
        self._library = {lora.filename: lora for lora in LORA_LIBRARY}
        self._refresh_available()

    def _refresh_available(self):
        """Check which LoRAs are actually available on disk."""
        self._available = set()
        lora_path = Path(self.lora_directory)
        if lora_path.exists():
            for f in lora_path.glob("*.safetensors"):
                self._available.add(f.name)

    def get_all(self) -> list[dict]:
        """Get all LoRAs in the library with availability status."""
        result = []
        for lora in LORA_LIBRARY:
            entry = lora.to_dict()
            entry["available"] = lora.filename in self._available
            result.append(entry)
        return result

    def get_by_category(self, category: str) -> list[dict]:
        """Get LoRAs filtered by category."""
        return [
            {**lora.to_dict(), "available": lora.filename in self._available}
            for lora in LORA_LIBRARY
            if lora.category == category
        ]

    def get_categories(self) -> list[str]:
        """Get list of all categories."""
        return list(set(lora.category for lora in LORA_LIBRARY))

    def detect_from_prompt(self, prompt: str, model: str = "illustrious") -> list[dict]:
        """
        Analyze a prompt and return suggested LoRAs.

        Returns LoRAs that:
        1. Have auto_enable=True and matching keywords
        2. Are compatible with the specified model
        3. Are available on disk
        """
        prompt_lower = prompt.lower()
        suggestions = []

        for lora in LORA_LIBRARY:
            # Skip if not available
            if lora.filename not in self._available:
                continue

            # Skip if not compatible with model
            if model not in lora.compatible_models and "sdxl" not in lora.compatible_models:
                continue

            # Check for keyword matches
            match_score = 0
            matched_keywords = []

            for keyword in lora.detection_keywords:
                if keyword.lower() in prompt_lower:
                    match_score += 1
                    matched_keywords.append(keyword)

            if match_score > 0:
                suggestions.append({
                    **lora.to_dict(),
                    "match_score": match_score,
                    "matched_keywords": matched_keywords,
                    "auto_enable": lora.auto_enable,
                })

        # Sort by match score and auto_enable preference
        suggestions.sort(key=lambda x: (-x["auto_enable"], -x["match_score"]))

        return suggestions

    def get_auto_loras(self, prompt: str, model: str = "illustrious") -> list[dict]:
        """
        Get LoRAs that should be automatically enabled based on prompt.

        Only returns LoRAs with auto_enable=True that have keyword matches.
        """
        suggestions = self.detect_from_prompt(prompt, model)
        return [
            {"name": s["filename"], "weight": s["default_weight"]}
            for s in suggestions
            if s["auto_enable"] and s["match_score"] > 0
        ]

    def get_by_filename(self, filename: str) -> Optional[dict]:
        """Get a specific LoRA by filename."""
        lora = self._library.get(filename)
        if lora:
            return {**lora.to_dict(), "available": filename in self._available}
        return None

    def is_available(self, filename: str) -> bool:
        """Check if a LoRA file exists."""
        return filename in self._available


# Global instance
_library_instance = None

def get_lora_library() -> LoRALibrary:
    """Get the global LoRA library instance."""
    global _library_instance
    if _library_instance is None:
        _library_instance = LoRALibrary()
    return _library_instance
