#!/usr/bin/env python3
"""
Test script for Grok 4.1 Fast extraction service.

This script tests the extraction service with a sample story transcript.
Run with: python test_grok_extraction.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.extraction import ExtractionService


# Sample transcript for testing
SAMPLE_TRANSCRIPT = """
Narrator: You wake up in a dimly lit tavern. The smell of ale and roasted meat fills the air.
Innkeeper: "Well, well. You're finally awake! You've been out cold for three hours."

You: "Where am I? What happened?"

Innkeeper: "You're at the Rusty Nail Inn, in the town of Millhaven. A merchant found you
unconscious on the road and brought you here. Said you were mumbling about shadows."

You: "I... I don't remember much. Just darkness and a feeling of dread."

Innkeeper: "That's the third traveler this week with the same story. Strange happenings
on the east road. The mayor's put out a bounty - 50 gold pieces for anyone who can
investigate and put a stop to whatever's causing it."

[DECISION POINT]
A) Accept the investigation job (50 gold reward)
B) Decline and rest at the inn for the night
C) Ask the innkeeper more questions about the incidents

You: "Tell me more about these incidents. What exactly happened to the other travelers?"

Innkeeper: "First was old Marcus the blacksmith. Tough as nails, that one. Found him
wandering in circles, completely disoriented. Second was a young mage named Elara.
She's still recovering upstairs - keeps talking about 'dark magic' and 'stolen memories.'"

[SKILL UNLOCKED: Perception Rank D]
You notice the innkeeper's hands are trembling slightly as he speaks.

[AFFECTION METER: Innkeeper - Trust: 15/100]

You: "I'd like to speak with this mage, Elara."

Innkeeper: "Room 7, upstairs. But be gentle - she's been through a lot."
"""


def main():
    """Run the extraction test."""
    print("=" * 60)
    print("Story Memory System - Grok 4.1 Fast Extraction Test")
    print("=" * 60)
    print()

    # Check for API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("❌ ERROR: XAI_API_KEY not found in environment variables")
        print("Please set your xAI API key in the .env file")
        print("Get your API key at: https://console.x.ai")
        return 1

    print("✓ Found xAI API key")
    print()

    # Initialize extraction service
    print("Initializing extraction service...")
    try:
        service = ExtractionService(api_key=api_key)
        print("✓ Extraction service initialized")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        return 1

    # Run extraction
    print("Extracting structured data from sample transcript...")
    print("(This may take 10-30 seconds)")
    print()

    try:
        extraction = service.extract_session(
            transcript=SAMPLE_TRANSCRIPT,
            story_premise="A fantasy RPG where mysterious incidents plague travelers",
            existing_entities=None,
            previous_summary=None,
        )

        print("✓ Extraction completed successfully!")
        print()
        print("=" * 60)
        print("EXTRACTION RESULTS")
        print("=" * 60)
        print()

        # Display results
        print(f"Session Summary:")
        print(f"  {extraction.session_summary}")
        print()

        print(f"Entities Extracted: {len(extraction.entities)}")
        for entity in extraction.entities:
            print(f"  - {entity.canonical_name} ({entity.entity_type})")
            print(f"    Aliases: {', '.join(entity.aliases) if entity.aliases else 'None'}")
            print(f"    Importance: {entity.importance}")
            if entity.initial_facts:
                print(f"    Facts: {len(entity.initial_facts)}")
            print()

        print(f"Events Extracted: {len(extraction.events)}")
        for event in extraction.events[:3]:  # Show first 3
            print(f"  - {event.description[:80]}...")
            print()

        print(f"Decisions: {len(extraction.decisions)}")
        for decision in extraction.decisions:
            print(f"  - Situation: {decision.situation[:60]}...")
            print(f"    Status: {decision.decision_type}")
            print()

        if extraction.protagonist_state:
            print("Protagonist State:")
            if extraction.protagonist_state.skills:
                print(f"  Skills: {len(extraction.protagonist_state.skills)}")
                for skill in extraction.protagonist_state.skills:
                    print(f"    - {skill.skill_name} (Rank: {skill.rank})")
            print()

        if extraction.character_states:
            print(f"Character Relationship States: {len(extraction.character_states)}")
            for state in extraction.character_states:
                print(f"  - {state.character_name}: {state.stat_type} = {state.current_value}/{state.max_value}")
            print()

        print("=" * 60)
        print("✓ TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
