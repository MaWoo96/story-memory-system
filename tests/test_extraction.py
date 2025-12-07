"""
Tests for the extraction service.
"""

import pytest
from schemas.extraction import (
    StoryExtraction,
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
)


def test_story_extraction_schema():
    """Test that StoryExtraction schema validates correctly."""
    # This is a basic schema validation test
    extraction_data = {
        "entities": [
            {
                "entity_type": "character",
                "canonical_name": "Test Character",
                "aliases": ["TC"],
                "description": "A test character",
                "facts": [],
                "importance": 0.8,
            }
        ],
        "relationships": [],
        "events": [],
        "decisions": [],
        "protagonist": {
            "stats": [],
            "skills": [],
            "inventory_changes": [],
            "status_effects": [],
        },
        "character_states": [],
        "world_state": {
            "current_time": None,
            "current_location": None,
            "pending_obligations": [],
        },
        "session_summary": "Test summary",
        "key_moments": ["Moment 1", "Moment 2"],
    }

    extraction = StoryExtraction(**extraction_data)
    assert extraction.entities[0].canonical_name == "Test Character"
    assert extraction.session_summary == "Test summary"


# TODO: Add more tests once extraction service is implemented
# @pytest.mark.asyncio
# async def test_extraction_service():
#     """Test extraction service with sample transcript."""
#     pass
