#!/usr/bin/env python3
"""
End-to-end test for the Story Memory System pipeline.

Tests the complete flow:
1. Create a story
2. Process a session transcript
3. Verify extraction and storage
4. Retrieve context for next session
5. Search memories

Run with: python test_pipeline.py
"""

import os
import sys
import asyncio
from uuid import UUID
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Sample transcript for testing
SAMPLE_TRANSCRIPT = """
Narrator: You wake up in a dimly lit tavern. The smell of ale and roasted meat fills the air.
A grizzled innkeeper with a scarred face polishes glasses behind the bar.

Innkeeper (Gareth): "Well, well. You're finally awake! You've been out cold for three hours."

You: "Where am I? What happened?"

Gareth: "You're at the Rusty Nail Inn, in the town of Millhaven. A merchant named
Marcus found you unconscious on the east road and brought you here. Said you were
mumbling about shadows and dark magic."

You: "I... I don't remember much. Just darkness and a feeling of dread."

Gareth: "That's the third traveler this week with the same story. Strange happenings
on the east road lately. The mayor's put out a bounty - 50 gold pieces for anyone
who can investigate and put a stop to whatever's causing it."

[DECISION POINT]
A) Accept the investigation job (50 gold reward)
B) Decline and rest at the inn for the night
C) Ask the innkeeper more questions about the incidents

You: "Tell me more about these incidents. What exactly happened to the other travelers?"

Gareth: "First was old Thomas the blacksmith. Tough as nails, that one. Found him
wandering in circles near the old watchtower, completely disoriented. Second was a
young mage named Elara from the Academy. She's still recovering upstairs - keeps
talking about 'dark magic' and 'stolen memories.' Poor girl was terrified."

[SKILL UNLOCKED: Perception - Rank D]
You notice the innkeeper's hands are trembling slightly as he speaks about the incidents.
There's fear in his eyes that he's trying to hide.

[AFFECTION METER: Gareth - Trust: 15/100 - "Wary but hopeful"]

You: "I'd like to speak with this mage, Elara. Perhaps she can tell me more."

Gareth: "Room 7, upstairs. But be gentle with her - she's been through a lot. And
here..." He slides a rusty iron key across the bar. "Take this. It's a spare key to
the room. Just in case."

[ITEM RECEIVED: Rusty Iron Key - "An old key to room 7 at the Rusty Nail Inn"]

You head upstairs and knock on room 7. A trembling voice answers.

Elara: "W-who's there? Go away!"

You: "I'm a traveler who experienced the same thing on the east road. I want to help."

After a long pause, the door creaks open. A young woman with disheveled silver hair
and dark circles under her eyes peers out at you.

Elara: "You... you survived the Shadow Walker too?"

[NEW ENTITY INTRODUCED: The Shadow Walker - mysterious entity attacking travelers]
"""


async def test_pipeline():
    """Run the complete pipeline test."""
    print("=" * 70)
    print("Story Memory System - Pipeline Test")
    print("=" * 70)
    print()

    # Check environment
    required = ["SUPABASE_URL", "SUPABASE_KEY", "XAI_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        return 1

    print("Environment configured")
    print()

    # Import services
    from api.dependencies import (
        get_supabase_client,
        get_extraction_service,
        get_storage_service,
        get_context_service,
    )

    db = get_supabase_client()
    extraction_service = get_extraction_service()
    storage_service = get_storage_service()
    context_service = get_context_service()

    print("Services initialized")
    print()

    # =========================================
    # Step 1: Create a test story
    # =========================================
    print("Step 1: Creating test story...")

    story_result = db.table("stories").insert({
        "user_id": "test-user-pipeline",
        "title": "The Shadow Walker Mystery",
        "premise": "A dark fantasy adventure where travelers on the east road are being attacked by a mysterious entity that steals memories.",
        "status": "active",
    }).execute()

    story = story_result.data[0]
    story_id = UUID(story["id"])
    print(f"  Created story: {story['title']} (ID: {story_id})")
    print()

    # =========================================
    # Step 2: Create a session and run extraction
    # =========================================
    print("Step 2: Creating session and running extraction...")
    print("  (This may take 10-30 seconds)")

    session_result = db.table("sessions").insert({
        "story_id": str(story_id),
        "session_number": 1,
        "transcript": SAMPLE_TRANSCRIPT,
        "status": "processing",
    }).execute()

    session = session_result.data[0]
    session_id = UUID(session["id"])
    print(f"  Created session #{session['session_number']} (ID: {session_id})")

    # Get existing entities (should be empty for first session)
    existing_entities = await storage_service.get_existing_entities(story_id)
    print(f"  Existing entities: {len(existing_entities)}")

    # Run extraction
    print("  Running Grok extraction...")
    extraction = extraction_service.extract_session(
        transcript=SAMPLE_TRANSCRIPT,
        story_premise=story["premise"],
        existing_entities=existing_entities,
        previous_summary=None,
    )
    print("  Extraction complete!")
    print()

    # =========================================
    # Step 3: Display extraction results
    # =========================================
    print("Step 3: Extraction Results")
    print("-" * 40)
    print(f"  Session Summary: {extraction.session_summary[:200]}...")
    print()
    print(f"  Entities: {len(extraction.entities)}")
    for entity in extraction.entities[:5]:
        print(f"    - {entity.canonical_name} ({entity.entity_type.value}) [importance: {entity.importance}]")
    if len(extraction.entities) > 5:
        print(f"    ... and {len(extraction.entities) - 5} more")
    print()

    print(f"  Events: {len(extraction.events)}")
    for event in extraction.events[:3]:
        print(f"    - [{event.event_type.value}] {event.description[:60]}...")
    print()

    print(f"  Decisions: {len(extraction.decisions)}")
    for decision in extraction.decisions:
        print(f"    - {decision.situation[:60]}... ({decision.decision_type.value})")
    print()

    if extraction.protagonist.skills:
        print(f"  Skills: {len(extraction.protagonist.skills)}")
        for skill in extraction.protagonist.skills:
            print(f"    - {skill.name} (Rank {skill.rank})")
    print()

    if extraction.character_states:
        print(f"  Character States: {len(extraction.character_states)}")
        for state in extraction.character_states:
            print(f"    - {state.character}: {state.stat_type.value} = {state.value}/{state.max}")
    print()

    # =========================================
    # Step 4: Store extraction results
    # =========================================
    print("Step 4: Storing extraction results...")

    stats = await storage_service.store_extraction(
        story_id=story_id,
        session_id=session_id,
        extraction=extraction,
    )

    print(f"  Entities created: {stats['entities_created']}")
    print(f"  Entities updated: {stats['entities_updated']}")
    print(f"  Facts created: {stats['facts_created']}")
    print(f"  Relationships created: {stats['relationships_created']}")
    print(f"  Events recorded: {stats['events_recorded']}")
    print(f"  Decisions captured: {stats['decisions_captured']}")
    print()

    # Update session status
    db.table("sessions").update({
        "status": "processed",
        "summary": extraction.session_summary,
        "key_moments": extraction.key_moments,
    }).eq("id", str(session_id)).execute()

    # =========================================
    # Step 5: Build context for next session
    # =========================================
    print("Step 5: Building context for next session...")

    context = await context_service.build_session_context(
        story_id=story_id,
        max_entities=10,
        recent_sessions=3,
    )

    print(f"  System context length: {len(context['system_context'])} chars")
    print(f"  Entity reference length: {len(context['entity_reference'])} chars")
    print(f"  Entities included: {len(context['entities'])}")
    print(f"  Pending decisions: {len(context['pending_decisions_data'])}")
    print()

    # Show a sample of the entity reference
    print("  Sample Entity Reference:")
    print("-" * 40)
    print(context['entity_reference'][:500])
    if len(context['entity_reference']) > 500:
        print("...")
    print()

    # =========================================
    # Step 6: Search memories
    # =========================================
    print("Step 6: Searching memories...")

    search_queries = ["Gareth", "shadow", "mage"]
    for query in search_queries:
        results = await context_service.search_memories(
            story_id=story_id,
            query=query,
            limit=3,
        )
        print(f"  Search '{query}': {len(results)} results")
        for r in results[:2]:
            print(f"    - [{r['type']}] {r['name']}: {r['content'][:50]}...")
    print()

    # =========================================
    # Step 7: Verify database state
    # =========================================
    print("Step 7: Verifying database state...")

    # Count entities
    entities = db.table("entities").select("id", count="exact").eq(
        "story_id", str(story_id)
    ).execute()
    print(f"  Entities in DB: {entities.count}")

    # Count facts
    facts = db.table("entity_facts").select("id", count="exact").execute()
    print(f"  Facts in DB: {facts.count}")

    # Count events
    events = db.table("events").select("id", count="exact").eq(
        "story_id", str(story_id)
    ).execute()
    print(f"  Events in DB: {events.count}")

    # Check protagonist state
    skills = db.table("protagonist_skills").select("skill_name, rank").eq(
        "story_id", str(story_id)
    ).execute()
    print(f"  Protagonist skills: {len(skills.data)}")
    for skill in skills.data:
        print(f"    - {skill['skill_name']} (Rank {skill['rank']})")

    inventory = db.table("protagonist_inventory").select("item_name").eq(
        "story_id", str(story_id)
    ).is_("lost_session_id", "null").execute()
    print(f"  Inventory items: {len(inventory.data)}")
    for item in inventory.data:
        print(f"    - {item['item_name']}")

    print()

    # =========================================
    # Cleanup (optional - comment out to keep data)
    # =========================================
    print("Step 8: Cleanup...")
    cleanup = input("  Delete test data? (y/N): ").strip().lower()
    if cleanup == 'y':
        db.table("stories").delete().eq("id", str(story_id)).execute()
        print("  Test data deleted")
    else:
        print(f"  Test data preserved (story_id: {story_id})")

    print()
    print("=" * 70)
    print("Pipeline test completed successfully!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(test_pipeline()))
