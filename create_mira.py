#!/usr/bin/env python3
import os
os.chdir('/Users/officemac/Projects/active/adhoc/misc/story-memory-system')

from dotenv import load_dotenv
load_dotenv('.env', override=True)

from supabase import create_client

client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
story_id = "5e033606-7113-491f-8691-390fd03b800c"

# Create Mira entity
entity = client.table("entities").insert({
    "story_id": story_id,
    "entity_type": "character",
    "canonical_name": "Mira",
    "description": "A young elven sorceress with flowing silver hair and violet eyes. She wears elegant blue robes adorned with arcane symbols.",
    "status": "active",
    "base_importance": 0.9,
    "computed_importance": 0.9,
    "metadata": {
        "race": "elf",
        "class": "sorceress",
        "age": "young adult"
    }
}).execute()

print(f"Created entity: {entity.data[0]['id']}")
print(f"Name: {entity.data[0]['canonical_name']}")

# Add some facts
entity_id = entity.data[0]['id']
facts = [
    {"entity_id": entity_id, "fact_type": "appearance", "fact_value": "Flowing silver hair that shimmers in moonlight"},
    {"entity_id": entity_id, "fact_type": "appearance", "fact_value": "Striking violet eyes that glow when casting spells"},
    {"entity_id": entity_id, "fact_type": "personality", "fact_value": "Curious and adventurous, always seeking ancient knowledge"},
    {"entity_id": entity_id, "fact_type": "ability", "fact_value": "Skilled in elemental magic, especially ice and lightning"},
]
client.table("entity_facts").insert(facts).execute()
print(f"Added {len(facts)} facts")

# Print entity ID for use
print(f"\n=== Use this entity_id for image generation ===")
print(f"Entity ID: {entity_id}")
