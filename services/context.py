"""
Context builder service.

Retrieves relevant memories and builds context for new sessions.
Implements importance-based entity selection, recency weighting,
and token budget management.
"""

from typing import Optional
from uuid import UUID
from datetime import datetime
import time


class ContextService:
    """Service for building context from stored memories."""

    def __init__(self, db_client):
        """
        Initialize context service.

        Args:
            db_client: Supabase client instance
        """
        self.db = db_client

    async def build_session_context(
        self,
        story_id: UUID,
        max_entities: int = 20,
        max_facts_per_entity: int = 5,
        recent_sessions: int = 3,
        include_all_pending: bool = True,
    ) -> dict:
        """
        Build context for starting a new session.

        Retrieves and prioritizes memories to fit within context limits.

        Args:
            story_id: Story UUID
            max_entities: Maximum entities to include
            max_facts_per_entity: Maximum facts per entity
            recent_sessions: Number of recent session summaries to include
            include_all_pending: Include all pending decisions regardless of limit

        Returns:
            Dictionary with structured context information
        """
        start_time = time.time()
        entities_included = []

        # 1. Get story info
        story = await self._get_story(story_id)
        if not story:
            raise ValueError(f"Story {story_id} not found")

        # 2. Get recent session summaries
        session_summaries = await self._get_recent_sessions(
            story_id, recent_sessions
        )

        # 3. Get top entities by computed importance
        entities = await self._get_top_entities(
            story_id, max_entities, max_facts_per_entity
        )
        entities_included = [e["id"] for e in entities]

        # 4. Get pending decisions
        pending_decisions = await self._get_pending_decisions(story_id)

        # 5. Get protagonist state
        protagonist_state = await self._get_protagonist_state(story_id)

        # 6. Get active character states (NPC relationship meters)
        character_states = await self._get_character_states(story_id)

        # 7. Get world state / pending obligations
        world_state = await self._get_world_state(story_id)

        # 8. Build formatted context strings
        context = self._format_context(
            story=story,
            session_summaries=session_summaries,
            entities=entities,
            pending_decisions=pending_decisions,
            protagonist_state=protagonist_state,
            character_states=character_states,
            world_state=world_state,
        )

        # Log query for debugging
        latency_ms = int((time.time() - start_time) * 1000)
        await self._log_memory_query(
            story_id=story_id,
            query_type="context_build",
            entities_included=entities_included,
            latency_ms=latency_ms,
        )

        return context

    async def _get_story(self, story_id: UUID) -> Optional[dict]:
        """Get story details."""
        result = self.db.table("stories").select(
            "id, title, premise, story_summary, current_situation, status"
        ).eq("id", str(story_id)).execute()

        return result.data[0] if result.data else None

    async def _get_recent_sessions(
        self,
        story_id: UUID,
        limit: int,
    ) -> list[dict]:
        """Get recent session summaries."""
        result = self.db.table("sessions").select(
            "session_number, summary, key_moments"
        ).eq("story_id", str(story_id)).eq(
            "status", "processed"
        ).order("session_number", desc=True).limit(limit).execute()

        # Reverse to chronological order
        return list(reversed(result.data)) if result.data else []

    async def _get_top_entities(
        self,
        story_id: UUID,
        limit: int,
        max_facts: int,
    ) -> list[dict]:
        """Get top entities by importance with their facts and relationships."""
        # Get entities
        entities_result = self.db.table("entities").select(
            "id, canonical_name, entity_type, description, status, computed_importance"
        ).eq("story_id", str(story_id)).neq(
            "status", "inactive"
        ).order("computed_importance", desc=True).limit(limit).execute()

        entities = []
        for entity in entities_result.data:
            entity_id = entity["id"]

            # Get valid facts
            facts_result = self.db.table("entity_facts").select(
                "fact_type, fact_value, importance"
            ).eq("entity_id", entity_id).is_(
                "invalidated_session_id", "null"
            ).order("importance", desc=True).limit(max_facts).execute()

            # Get active relationships where this entity is source
            outgoing_rels = self.db.table("entity_relationships").select(
                "target_entity_id, relationship_type, description"
            ).eq("source_entity_id", entity_id).is_(
                "ended_session_id", "null"
            ).execute()

            # Resolve target entity names
            relationships = []
            for rel in outgoing_rels.data:
                target = self.db.table("entities").select(
                    "canonical_name"
                ).eq("id", rel["target_entity_id"]).execute()

                if target.data:
                    relationships.append({
                        "target": target.data[0]["canonical_name"],
                        "type": rel["relationship_type"],
                        "description": rel["description"],
                    })

            entities.append({
                "id": entity_id,
                "name": entity["canonical_name"],
                "type": entity["entity_type"],
                "description": entity["description"],
                "status": entity["status"],
                "importance": entity["computed_importance"],
                "facts": facts_result.data,
                "relationships": relationships,
            })

        return entities

    async def _get_pending_decisions(self, story_id: UUID) -> list[dict]:
        """Get all pending decisions."""
        result = self.db.table("decisions").select(
            "id, situation, alternatives, importance"
        ).eq("story_id", str(story_id)).eq(
            "decision_type", "pending"
        ).order("importance", desc=True).execute()

        return result.data if result.data else []

    async def _get_protagonist_state(self, story_id: UUID) -> dict:
        """Get current protagonist state."""
        # Stats
        stats_result = self.db.table("protagonist_stats").select(
            "stat_name, current_value, max_value, stat_type"
        ).eq("story_id", str(story_id)).execute()

        # Skills
        skills_result = self.db.table("protagonist_skills").select(
            "skill_name, rank, description, mechanical_effect"
        ).eq("story_id", str(story_id)).execute()

        # Active inventory
        inventory_result = self.db.table("protagonist_inventory").select(
            "item_name, description, properties, equipped"
        ).eq("story_id", str(story_id)).is_(
            "lost_session_id", "null"
        ).execute()

        # Active status effects
        effects_result = self.db.table("protagonist_status_effects").select(
            "effect_name, description, is_temporary"
        ).eq("story_id", str(story_id)).is_(
            "removed_session_id", "null"
        ).execute()

        return {
            "stats": stats_result.data or [],
            "skills": skills_result.data or [],
            "inventory": inventory_result.data or [],
            "status_effects": effects_result.data or [],
        }

    async def _get_character_states(self, story_id: UUID) -> list[dict]:
        """Get NPC relationship meters."""
        result = self.db.table("character_states").select(
            "entity_id, stat_type, current_value, max_value, label"
        ).eq("story_id", str(story_id)).execute()

        states = []
        for state in result.data:
            # Get entity name
            entity = self.db.table("entities").select(
                "canonical_name"
            ).eq("id", state["entity_id"]).execute()

            if entity.data:
                states.append({
                    "character": entity.data[0]["canonical_name"],
                    "stat_type": state["stat_type"],
                    "value": state["current_value"],
                    "max": state["max_value"],
                    "label": state["label"],
                })

        return states

    async def _get_world_state(self, story_id: UUID) -> dict:
        """Get world state from story and recent decisions."""
        story = self.db.table("stories").select(
            "current_situation"
        ).eq("id", str(story_id)).execute()

        # Get pending obligations from recent pending decisions
        obligations = self.db.table("decisions").select(
            "situation"
        ).eq("story_id", str(story_id)).eq(
            "decision_type", "pending"
        ).execute()

        return {
            "current_situation": story.data[0]["current_situation"] if story.data else None,
            "pending_obligations": [d["situation"] for d in obligations.data] if obligations.data else [],
        }

    def _format_context(
        self,
        story: dict,
        session_summaries: list[dict],
        entities: list[dict],
        pending_decisions: list[dict],
        protagonist_state: dict,
        character_states: list[dict],
        world_state: dict,
    ) -> dict:
        """Format context into strings for LLM injection."""

        # Build system context string
        system_parts = []

        # Story premise and summary
        if story.get("premise"):
            system_parts.append(f"STORY PREMISE:\n{story['premise']}")

        if story.get("story_summary"):
            system_parts.append(f"STORY SO FAR:\n{story['story_summary']}")

        # Recent sessions
        if session_summaries:
            recent = "\n\n".join([
                f"Session {s['session_number']}:\n{s['summary']}"
                for s in session_summaries
            ])
            system_parts.append(f"RECENT SESSIONS:\n{recent}")

        # Current situation
        if world_state.get("current_situation"):
            system_parts.append(f"CURRENT SITUATION:\n{world_state['current_situation']}")

        system_context = "\n\n---\n\n".join(system_parts)

        # Build entity reference
        entity_parts = []
        for entity in entities:
            entity_str = f"**{entity['name']}** ({entity['type']})"
            if entity.get("status") != "active":
                entity_str += f" [{entity['status']}]"
            entity_str += f"\n{entity['description']}"

            if entity.get("facts"):
                facts = [f"- {f['fact_value']}" for f in entity["facts"]]
                entity_str += "\nFacts:\n" + "\n".join(facts)

            if entity.get("relationships"):
                rels = [f"- {r['type']} → {r['target']}: {r['description']}"
                        for r in entity["relationships"]]
                entity_str += "\nRelationships:\n" + "\n".join(rels)

            entity_parts.append(entity_str)

        entity_reference = "\n\n".join(entity_parts) if entity_parts else "No entities recorded yet."

        # Build protagonist state string
        protag_parts = []

        if protagonist_state.get("stats"):
            stats = [f"- {s['stat_name']}: {s['current_value']}"
                     + (f"/{s['max_value']}" if s.get('max_value') else "")
                     for s in protagonist_state["stats"]]
            protag_parts.append("STATS:\n" + "\n".join(stats))

        if protagonist_state.get("skills"):
            skills = [f"- {s['skill_name']}"
                      + (f" (Rank {s['rank']})" if s.get('rank') else "")
                      + f": {s['description']}"
                      for s in protagonist_state["skills"]]
            protag_parts.append("SKILLS:\n" + "\n".join(skills))

        if protagonist_state.get("inventory"):
            equipped = [i for i in protagonist_state["inventory"] if i.get("equipped")]
            carried = [i for i in protagonist_state["inventory"] if not i.get("equipped")]

            if equipped:
                items = [f"- {i['item_name']}" + (f": {i['description']}" if i.get('description') else "")
                         for i in equipped]
                protag_parts.append("EQUIPPED:\n" + "\n".join(items))

            if carried:
                items = [f"- {i['item_name']}" for i in carried]
                protag_parts.append("INVENTORY:\n" + "\n".join(items))

        if protagonist_state.get("status_effects"):
            effects = [f"- {e['effect_name']}: {e['description']}"
                       for e in protagonist_state["status_effects"]]
            protag_parts.append("STATUS EFFECTS:\n" + "\n".join(effects))

        protagonist_context = "\n\n".join(protag_parts) if protag_parts else ""

        # Build relationship meters string
        if character_states:
            meters = [f"- {cs['character']} ({cs['stat_type']}): {cs['value']}/{cs['max']}"
                      + (f" - \"{cs['label']}\"" if cs.get('label') else "")
                      for cs in character_states]
            relationship_meters = "NPC RELATIONSHIP METERS:\n" + "\n".join(meters)
        else:
            relationship_meters = ""

        # Build pending decisions string
        if pending_decisions:
            decisions = [f"- {d['situation']}\n  Options: {', '.join(d['alternatives'] or [])}"
                         for d in pending_decisions]
            pending_context = "PENDING DECISIONS:\n" + "\n".join(decisions)
        else:
            pending_context = ""

        return {
            "system_context": system_context,
            "story_summary": story.get("story_summary", ""),
            "recent_sessions": session_summaries,
            "entity_reference": entity_reference,
            "entities": entities,
            "protagonist_state": protagonist_context,
            "protagonist_data": protagonist_state,
            "relationship_meters": relationship_meters,
            "character_states": character_states,
            "pending_decisions": pending_context,
            "pending_decisions_data": pending_decisions,
            "world_state": world_state,
        }

    async def search_memories(
        self,
        story_id: UUID,
        query: str,
        limit: int = 10,
        entity_types: Optional[list[str]] = None,
        include_facts: bool = True,
        include_events: bool = True,
    ) -> list[dict]:
        """
        Search story memories for specific facts.

        Uses text search on entity descriptions, facts, and events.

        Args:
            story_id: Story UUID
            query: Search query
            limit: Maximum results
            entity_types: Filter by entity types
            include_facts: Include fact search
            include_events: Include event search

        Returns:
            List of matching memories
        """
        start_time = time.time()
        results = []
        query_lower = query.lower()

        # Search entities
        entity_query = self.db.table("entities").select(
            "id, canonical_name, entity_type, description, computed_importance"
        ).eq("story_id", str(story_id)).neq("status", "inactive")

        if entity_types:
            entity_query = entity_query.in_("entity_type", entity_types)

        entities = entity_query.execute()

        for entity in entities.data:
            # Check if query matches name or description
            if (query_lower in entity["canonical_name"].lower() or
                (entity.get("description") and query_lower in entity["description"].lower())):
                results.append({
                    "type": "entity",
                    "entity_type": entity["entity_type"],
                    "name": entity["canonical_name"],
                    "content": entity["description"],
                    "importance": entity["computed_importance"],
                    "id": entity["id"],
                })

        # Search facts
        if include_facts:
            facts = self.db.table("entity_facts").select(
                "entity_id, fact_type, fact_value, importance"
            ).ilike("fact_value", f"%{query}%").is_(
                "invalidated_session_id", "null"
            ).execute()

            for fact in facts.data:
                # Get entity info
                entity = self.db.table("entities").select(
                    "canonical_name, story_id"
                ).eq("id", fact["entity_id"]).execute()

                if entity.data and entity.data[0]["story_id"] == str(story_id):
                    results.append({
                        "type": "fact",
                        "entity_type": None,
                        "name": entity.data[0]["canonical_name"],
                        "content": fact["fact_value"],
                        "importance": fact["importance"],
                        "fact_type": fact["fact_type"],
                    })

        # Search events
        if include_events:
            events = self.db.table("events").select(
                "event_type, description, importance"
            ).eq("story_id", str(story_id)).ilike(
                "description", f"%{query}%"
            ).execute()

            for event in events.data:
                results.append({
                    "type": "event",
                    "entity_type": None,
                    "name": event["event_type"],
                    "content": event["description"],
                    "importance": event["importance"],
                })

        # Sort by importance and limit
        results.sort(key=lambda x: x["importance"], reverse=True)
        results = results[:limit]

        # Log query
        latency_ms = int((time.time() - start_time) * 1000)
        await self._log_memory_query(
            story_id=story_id,
            query_type="search",
            latency_ms=latency_ms,
        )

        return results

    async def get_entity_detail(
        self,
        entity_id: UUID,
    ) -> Optional[dict]:
        """Get full details for a specific entity."""
        # Get entity
        entity = self.db.table("entities").select("*").eq(
            "id", str(entity_id)
        ).execute()

        if not entity.data:
            return None

        entity = entity.data[0]

        # Get all aliases
        aliases = self.db.table("entity_aliases").select("alias").eq(
            "entity_id", str(entity_id)
        ).execute()

        # Get all facts (including invalidated for history)
        facts = self.db.table("entity_facts").select("*").eq(
            "entity_id", str(entity_id)
        ).order("created_at").execute()

        # Get relationships
        outgoing = self.db.table("entity_relationships").select("*").eq(
            "source_entity_id", str(entity_id)
        ).execute()

        incoming = self.db.table("entity_relationships").select("*").eq(
            "target_entity_id", str(entity_id)
        ).execute()

        # Get event participations
        participations = self.db.table("event_participants").select(
            "event_id"
        ).eq("entity_id", str(entity_id)).execute()

        event_ids = [p["event_id"] for p in participations.data]
        events = []
        if event_ids:
            events_result = self.db.table("events").select("*").in_(
                "id", event_ids
            ).order("sequence_in_session").execute()
            events = events_result.data

        return {
            "entity": entity,
            "aliases": [a["alias"] for a in aliases.data],
            "facts": {
                "current": [f for f in facts.data if not f.get("invalidated_session_id")],
                "invalidated": [f for f in facts.data if f.get("invalidated_session_id")],
            },
            "relationships": {
                "outgoing": outgoing.data,
                "incoming": incoming.data,
            },
            "events": events,
        }

    async def get_story_timeline(
        self,
        story_id: UUID,
        start_session: Optional[int] = None,
        end_session: Optional[int] = None,
        event_types: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get chronological timeline of events."""
        # Get sessions in range
        sessions_query = self.db.table("sessions").select(
            "id, session_number"
        ).eq("story_id", str(story_id)).eq("status", "processed")

        if start_session is not None:
            sessions_query = sessions_query.gte("session_number", start_session)
        if end_session is not None:
            sessions_query = sessions_query.lte("session_number", end_session)

        sessions = sessions_query.order("session_number").execute()
        session_map = {s["id"]: s["session_number"] for s in sessions.data}
        session_ids = list(session_map.keys())

        if not session_ids:
            return []

        # Get events
        events_query = self.db.table("events").select(
            "id, session_id, event_type, description, sequence_in_session, importance"
        ).in_("session_id", session_ids)

        if event_types:
            events_query = events_query.in_("event_type", event_types)

        events = events_query.order("created_at").limit(limit).execute()

        # Enrich with session number and participants
        timeline = []
        for event in events.data:
            # Get participants
            participants = self.db.table("event_participants").select(
                "entity_id"
            ).eq("event_id", event["id"]).execute()

            participant_names = []
            for p in participants.data:
                entity = self.db.table("entities").select(
                    "canonical_name"
                ).eq("id", p["entity_id"]).execute()
                if entity.data:
                    participant_names.append(entity.data[0]["canonical_name"])

            timeline.append({
                "session_number": session_map.get(event["session_id"]),
                "sequence": event["sequence_in_session"],
                "event_type": event["event_type"],
                "description": event["description"],
                "importance": event["importance"],
                "participants": participant_names,
            })

        # Sort by session then sequence
        timeline.sort(key=lambda x: (x["session_number"] or 0, x["sequence"] or 0))

        return timeline

    async def _log_memory_query(
        self,
        story_id: UUID,
        query_type: str,
        entities_included: Optional[list[str]] = None,
        latency_ms: int = 0,
    ) -> None:
        """Log memory query for debugging and optimization."""
        try:
            self.db.table("memory_queries").insert({
                "story_id": str(story_id),
                "query_type": query_type,
                "entities_included": entities_included,
                "latency_ms": latency_ms,
            }).execute()
        except Exception:
            # Table might not exist yet, ignore
            pass

    # ============================================
    # SEMANTIC MEMORY RETRIEVAL (SpicyChat Pattern)
    # ============================================

    async def get_semantic_memories(
        self,
        story_id: UUID,
        characters_present: Optional[list[str]] = None,
        current_emotion: Optional[str] = None,
        current_topics: Optional[list[str]] = None,
        max_memories: int = 20,
        include_pinned: bool = True,
    ) -> list[dict]:
        """
        Retrieve semantic memories with relevance scoring.

        Based on SpicyChat's Semantic Memory 2.0 pattern:
        - Always include pinned memories
        - Score by recency, importance, character overlap, emotion match
        - Return top N most relevant memories

        Args:
            story_id: Story UUID
            characters_present: Characters in current scene for overlap scoring
            current_emotion: Current scene emotion for matching
            current_topics: Current scene topics for matching
            max_memories: Maximum memories to return
            include_pinned: Always include pinned memories

        Returns:
            List of scored and sorted semantic memories
        """
        # Get all non-hidden memories
        result = self.db.table("semantic_memories").select("*").eq(
            "story_id", str(story_id)
        ).eq("hidden", False).execute()

        memories = result.data or []

        # Separate pinned and regular memories
        pinned = [m for m in memories if m.get("pinned")] if include_pinned else []
        regular = [m for m in memories if not m.get("pinned")]

        # Score regular memories
        scored_regular = []
        for memory in regular:
            score = self._calculate_memory_score(
                memory=memory,
                characters_present=characters_present,
                current_emotion=current_emotion,
                current_topics=current_topics,
            )
            memory["_relevance_score"] = score
            scored_regular.append(memory)

        # Sort by score
        scored_regular.sort(key=lambda m: m["_relevance_score"], reverse=True)

        # Take top N, reserving space for pinned
        available_slots = max_memories - len(pinned)
        top_regular = scored_regular[:available_slots]

        # Combine pinned + top regular
        result_memories = pinned + top_regular

        # Update access counts for retrieved memories
        memory_ids = [m["id"] for m in result_memories]
        if memory_ids:
            try:
                for mid in memory_ids:
                    self.db.table("semantic_memories").update({
                        "access_count": self.db.rpc("increment", {"x": 1}),
                        "last_accessed_at": datetime.utcnow().isoformat(),
                    }).eq("id", mid).execute()
            except Exception:
                pass  # Non-critical update

        return result_memories

    def _calculate_memory_score(
        self,
        memory: dict,
        characters_present: Optional[list[str]] = None,
        current_emotion: Optional[str] = None,
        current_topics: Optional[list[str]] = None,
    ) -> float:
        """
        Calculate relevance score for a memory.

        Score components (weights sum to 1.0):
        - Recency: 0.15 (time decay)
        - Importance: 0.25 (base importance)
        - Character overlap: 0.25 (matching characters)
        - Emotion match: 0.15 (matching emotion)
        - Topic match: 0.15 (matching topics)
        - Setup/payoff: 0.05 (narrative continuity bonus)

        Returns:
            Float score 0.0-1.0 (can exceed 1.0 with bonuses)
        """
        score = 0.0

        # 1. Recency score (decay factor 0.995 per day)
        created_at = memory.get("created_at")
        if created_at:
            try:
                if isinstance(created_at, str):
                    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    created = created_at
                days_ago = (datetime.now(created.tzinfo) - created).days
                recency_score = pow(0.995, days_ago)
            except Exception:
                recency_score = 0.5
        else:
            recency_score = 0.5
        score += 0.15 * recency_score

        # 2. Importance score
        importance = memory.get("importance", 0.5)
        score += 0.25 * importance

        # 3. Character overlap score
        if characters_present:
            memory_chars = memory.get("characters_involved") or []
            if isinstance(memory_chars, str):
                import json
                try:
                    memory_chars = json.loads(memory_chars)
                except Exception:
                    memory_chars = []

            overlap = len(set(memory_chars) & set(characters_present))
            max_possible = max(len(memory_chars), len(characters_present), 1)
            char_score = overlap / max_possible
        else:
            char_score = 0.5  # Neutral if no context provided
        score += 0.25 * char_score

        # 4. Emotion match score
        if current_emotion:
            memory_emotion = memory.get("primary_emotion", "")
            emotion_match = 1.0 if memory_emotion == current_emotion else 0.0

            # Partial credit for related emotions
            emotion_groups = {
                "positive": ["joy", "excitement", "trust", "tenderness", "surprise"],
                "negative": ["tension", "conflict", "fear", "sadness", "anger"],
                "intimate": ["intimacy", "lust", "tenderness", "trust"],
            }
            if emotion_match < 1.0:
                for group in emotion_groups.values():
                    if memory_emotion in group and current_emotion in group:
                        emotion_match = 0.5
                        break
        else:
            emotion_match = 0.5
        score += 0.15 * emotion_match

        # 5. Topic match score
        if current_topics:
            memory_topics = memory.get("topics") or []
            if isinstance(memory_topics, str):
                import json
                try:
                    memory_topics = json.loads(memory_topics)
                except Exception:
                    memory_topics = []

            if memory_topics:
                overlap = len(set(memory_topics) & set(current_topics))
                topic_score = overlap / len(current_topics)
            else:
                topic_score = 0.0
        else:
            topic_score = 0.5
        score += 0.15 * topic_score

        # 6. Setup/payoff bonus
        if memory.get("setup_for_payoff"):
            score += 0.05

        return score

    async def get_relevant_memories_for_chat(
        self,
        story_id: UUID,
        recent_messages: list[dict],
        max_memories: int = 10,
    ) -> list[dict]:
        """
        Get relevant semantic memories based on recent chat context.

        Extracts characters and themes from recent messages to find relevant memories.

        Args:
            story_id: Story UUID
            recent_messages: Recent chat messages for context extraction
            max_memories: Maximum memories to return

        Returns:
            List of relevant memories with scores
        """
        # Extract characters mentioned in recent messages
        characters_mentioned = set()
        content_text = ""

        for msg in recent_messages[-10:]:  # Look at last 10 messages
            content = msg.get("content", "")
            content_text += " " + content

            # Simple character extraction (could use NER in future)
            # For now, get from database and check mentions
            entities = self.db.table("entities").select(
                "canonical_name"
            ).eq("story_id", str(story_id)).eq(
                "entity_type", "character"
            ).execute()

            for entity in entities.data:
                name = entity["canonical_name"]
                if name.lower() in content.lower():
                    characters_mentioned.add(name)

        # Infer emotion from content (simple keyword matching)
        emotion_keywords = {
            "joy": ["happy", "joy", "excited", "wonderful", "amazing"],
            "tension": ["worried", "nervous", "tense", "afraid", "anxious"],
            "intimacy": ["close", "intimate", "warm", "gentle", "tender"],
            "conflict": ["angry", "furious", "fight", "argue", "conflict"],
            "fear": ["scared", "terrified", "fear", "horror"],
            "sadness": ["sad", "crying", "tears", "grief", "loss"],
        }

        current_emotion = None
        content_lower = content_text.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(kw in content_lower for kw in keywords):
                current_emotion = emotion
                break

        # Get memories with scoring
        memories = await self.get_semantic_memories(
            story_id=story_id,
            characters_present=list(characters_mentioned),
            current_emotion=current_emotion,
            max_memories=max_memories,
        )

        return memories

    def format_memories_for_context(
        self,
        memories: list[dict],
        include_scores: bool = False,
    ) -> str:
        """
        Format semantic memories into a context string for LLM injection.

        Args:
            memories: List of semantic memory dicts
            include_scores: Include relevance scores (for debugging)

        Returns:
            Formatted string ready for LLM context
        """
        if not memories:
            return ""

        parts = ["IMPORTANT MEMORIES:"]

        for memory in memories:
            chars = memory.get("characters_involved", [])
            if isinstance(chars, str):
                import json
                try:
                    chars = json.loads(chars)
                except Exception:
                    chars = []

            char_str = f" [{', '.join(chars)}]" if chars else ""

            # Format based on pinned status
            prefix = "📌" if memory.get("pinned") else "-"
            text = memory.get("memory_text", "")

            if include_scores:
                score = memory.get("_relevance_score", 0)
                parts.append(f"{prefix}{char_str} {text} (score: {score:.2f})")
            else:
                parts.append(f"{prefix}{char_str} {text}")

        return "\n".join(parts)
