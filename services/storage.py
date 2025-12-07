"""
Memory storage service.

Handles storing extracted memories to the database with entity resolution,
temporal fact handling, and game state updates.
"""

from typing import Optional
from uuid import UUID
from datetime import datetime

from schemas.extraction import (
    StoryExtraction,
    Entity,
    Relationship,
    Event,
    Decision,
    Protagonist,
    CharacterState,
    InventoryAction,
)


class StorageService:
    """Service for storing extracted memories in the database."""

    def __init__(self, db_client):
        """
        Initialize storage service.

        Args:
            db_client: Supabase client instance
        """
        self.db = db_client

    async def store_extraction(
        self,
        story_id: UUID,
        session_id: UUID,
        extraction: StoryExtraction,
    ) -> dict:
        """
        Store extracted memory data to the database.

        Args:
            story_id: Story UUID
            session_id: Session UUID
            extraction: Extracted memory data

        Returns:
            Dictionary with storage statistics
        """
        stats = {
            "entities_created": 0,
            "entities_updated": 0,
            "facts_created": 0,
            "relationships_created": 0,
            "events_recorded": 0,
            "decisions_captured": 0,
        }

        # Build entity name -> ID mapping for relationship resolution
        entity_map: dict[str, UUID] = {}

        # 1. Store entities (with alias matching)
        for entity in extraction.entities:
            entity_id, is_new = await self._store_entity(
                story_id, session_id, entity
            )
            entity_map[entity.canonical_name.lower()] = entity_id
            for alias in entity.aliases:
                entity_map[alias.lower()] = entity_id

            if is_new:
                stats["entities_created"] += 1
            else:
                stats["entities_updated"] += 1

            # Store facts for this entity
            facts_created = await self._store_entity_facts(
                entity_id, session_id, entity.facts
            )
            stats["facts_created"] += facts_created

        # 2. Store relationships
        for rel in extraction.relationships:
            created = await self._store_relationship(
                story_id, session_id, rel, entity_map
            )
            if created:
                stats["relationships_created"] += 1

        # 3. Store events
        for idx, event in enumerate(extraction.events):
            await self._store_event(
                story_id, session_id, event, idx, entity_map
            )
            stats["events_recorded"] += 1

        # 4. Store decisions
        for decision in extraction.decisions:
            await self._store_decision(story_id, session_id, decision)
            stats["decisions_captured"] += 1

        # 5. Update protagonist state
        await self._update_protagonist_state(
            story_id, session_id, extraction.protagonist
        )

        # 6. Update character states (NPC relationship meters)
        for char_state in extraction.character_states:
            await self._update_character_state(
                story_id, session_id, char_state, entity_map
            )

        # 7. Update session with summary and key moments
        await self._update_session(
            session_id,
            extraction.session_summary,
            extraction.key_moments,
            extraction.model_dump(),
        )

        # 8. Update world state on story
        await self._update_world_state(story_id, extraction.world_state)

        return stats

    async def _store_entity(
        self,
        story_id: UUID,
        session_id: UUID,
        entity: Entity,
    ) -> tuple[UUID, bool]:
        """
        Store or update an entity, handling alias resolution.

        Returns:
            Tuple of (entity_id, is_new)
        """
        # First, try to find existing entity by canonical name or alias
        existing = await self._find_entity_by_name_or_alias(
            story_id, entity.canonical_name
        )

        if existing:
            # Update existing entity
            entity_id = existing["id"]

            # Update description and importance if this mention is more important
            update_data = {
                "last_session_id": str(session_id),
                "mention_count": existing["mention_count"] + 1,
                "updated_at": datetime.utcnow().isoformat(),
            }

            # Update description if new one is longer/more detailed
            if entity.description and (
                not existing.get("description") or
                len(entity.description) > len(existing.get("description", ""))
            ):
                update_data["description"] = entity.description

            # Update base importance if higher
            if entity.importance > existing.get("base_importance", 0):
                update_data["base_importance"] = entity.importance

            self.db.table("entities").update(update_data).eq(
                "id", str(entity_id)
            ).execute()

            # Add any new aliases
            await self._add_entity_aliases(entity_id, entity.aliases)

            return UUID(entity_id), False
        else:
            # Create new entity
            result = self.db.table("entities").insert({
                "story_id": str(story_id),
                "entity_type": entity.entity_type.value,
                "canonical_name": entity.canonical_name,
                "description": entity.description,
                "first_session_id": str(session_id),
                "last_session_id": str(session_id),
                "base_importance": entity.importance,
                "computed_importance": entity.importance,
            }).execute()

            entity_id = UUID(result.data[0]["id"])

            # Add aliases
            await self._add_entity_aliases(entity_id, entity.aliases)
            # Also add canonical name as an alias for easier lookup
            await self._add_entity_aliases(entity_id, [entity.canonical_name])

            return entity_id, True

    async def _find_entity_by_name_or_alias(
        self,
        story_id: UUID,
        name: str,
    ) -> Optional[dict]:
        """Find an entity by canonical name or alias."""
        name_lower = name.lower()

        # First check canonical name
        result = self.db.table("entities").select("*").eq(
            "story_id", str(story_id)
        ).ilike("canonical_name", name_lower).execute()

        if result.data:
            return result.data[0]

        # Check aliases
        alias_result = self.db.table("entity_aliases").select(
            "entity_id"
        ).eq("alias_lower", name_lower).execute()

        if alias_result.data:
            entity_id = alias_result.data[0]["entity_id"]
            # Verify this entity belongs to the story
            entity_result = self.db.table("entities").select("*").eq(
                "id", entity_id
            ).eq("story_id", str(story_id)).execute()

            if entity_result.data:
                return entity_result.data[0]

        return None

    async def _add_entity_aliases(
        self,
        entity_id: UUID,
        aliases: list[str],
    ) -> None:
        """Add aliases to an entity, skipping duplicates."""
        for alias in aliases:
            if not alias:
                continue
            try:
                self.db.table("entity_aliases").insert({
                    "entity_id": str(entity_id),
                    "alias": alias,
                    "alias_lower": alias.lower(),
                }).execute()
            except Exception:
                # Alias already exists, skip
                pass

    async def _store_entity_facts(
        self,
        entity_id: UUID,
        session_id: UUID,
        facts: list,
    ) -> int:
        """Store facts for an entity, handling temporal truth."""
        created = 0
        for fact in facts:
            # Check if this fact already exists
            existing = self.db.table("entity_facts").select("id").eq(
                "entity_id", str(entity_id)
            ).eq("fact_type", fact.fact_type.value).eq(
                "fact_value", fact.fact_value
            ).is_("invalidated_session_id", "null").execute()

            if not existing.data:
                # New fact
                self.db.table("entity_facts").insert({
                    "entity_id": str(entity_id),
                    "fact_type": fact.fact_type.value,
                    "fact_value": fact.fact_value,
                    "established_session_id": str(session_id),
                    "importance": fact.importance,
                }).execute()
                created += 1

        return created

    async def _store_relationship(
        self,
        story_id: UUID,
        session_id: UUID,
        rel: Relationship,
        entity_map: dict[str, UUID],
    ) -> bool:
        """Store a relationship between entities."""
        # Resolve entity IDs
        source_id = entity_map.get(rel.source.lower())
        target_id = entity_map.get(rel.target.lower())

        if not source_id or not target_id:
            # Try to find entities by name
            if not source_id:
                source_entity = await self._find_entity_by_name_or_alias(
                    story_id, rel.source
                )
                if source_entity:
                    source_id = UUID(source_entity["id"])

            if not target_id:
                target_entity = await self._find_entity_by_name_or_alias(
                    story_id, rel.target
                )
                if target_entity:
                    target_id = UUID(target_entity["id"])

        if not source_id or not target_id:
            # Can't create relationship without both entities
            return False

        # Check if relationship already exists
        existing = self.db.table("entity_relationships").select("id").eq(
            "source_entity_id", str(source_id)
        ).eq("target_entity_id", str(target_id)).eq(
            "relationship_type", rel.relationship_type.value
        ).is_("ended_session_id", "null").execute()

        if existing.data:
            # Update existing relationship
            self.db.table("entity_relationships").update({
                "description": rel.description,
                "importance": rel.importance,
                "updated_at": datetime.utcnow().isoformat(),
            }).eq("id", existing.data[0]["id"]).execute()
            return False
        else:
            # Create new relationship
            self.db.table("entity_relationships").insert({
                "story_id": str(story_id),
                "source_entity_id": str(source_id),
                "target_entity_id": str(target_id),
                "relationship_type": rel.relationship_type.value,
                "description": rel.description,
                "established_session_id": str(session_id),
                "importance": rel.importance,
            }).execute()
            return True

    async def _store_event(
        self,
        story_id: UUID,
        session_id: UUID,
        event: Event,
        sequence: int,
        entity_map: dict[str, UUID],
    ) -> UUID:
        """Store an event and link participants."""
        result = self.db.table("events").insert({
            "story_id": str(story_id),
            "session_id": str(session_id),
            "event_type": event.event_type.value,
            "description": event.description,
            "sequence_in_session": sequence,
            "importance": event.importance,
        }).execute()

        event_id = UUID(result.data[0]["id"])

        # Link participants
        for participant in event.participants:
            entity_id = entity_map.get(participant.lower())
            if not entity_id:
                entity = await self._find_entity_by_name_or_alias(
                    story_id, participant
                )
                if entity:
                    entity_id = UUID(entity["id"])

            if entity_id:
                try:
                    self.db.table("event_participants").insert({
                        "event_id": str(event_id),
                        "entity_id": str(entity_id),
                    }).execute()
                except Exception:
                    pass  # Duplicate, skip

        return event_id

    async def _store_decision(
        self,
        story_id: UUID,
        session_id: UUID,
        decision: Decision,
    ) -> UUID:
        """Store a decision point."""
        result = self.db.table("decisions").insert({
            "story_id": str(story_id),
            "session_id": str(session_id),
            "situation": decision.situation,
            "choice_made": decision.choice_made,
            "alternatives": decision.options,
            "decision_type": decision.decision_type.value,
            "importance": decision.importance,
        }).execute()

        return UUID(result.data[0]["id"])

    async def _update_protagonist_state(
        self,
        story_id: UUID,
        session_id: UUID,
        protagonist: Protagonist,
    ) -> None:
        """Update protagonist stats, skills, inventory, and status effects."""

        # Update stats
        for stat in protagonist.stats:
            existing = self.db.table("protagonist_stats").select("id").eq(
                "story_id", str(story_id)
            ).eq("stat_name", stat.stat).execute()

            stat_data = {
                "story_id": str(story_id),
                "stat_name": stat.stat,
                "current_value": str(stat.value) if stat.value is not None else None,
                "max_value": str(stat.max) if stat.max is not None else None,
                "stat_type": stat.type.value,
                "updated_at": datetime.utcnow().isoformat(),
            }

            if existing.data:
                self.db.table("protagonist_stats").update(stat_data).eq(
                    "id", existing.data[0]["id"]
                ).execute()
            else:
                self.db.table("protagonist_stats").insert(stat_data).execute()

        # Update skills
        for skill in protagonist.skills:
            existing = self.db.table("protagonist_skills").select("id").eq(
                "story_id", str(story_id)
            ).eq("skill_name", skill.name).execute()

            skill_data = {
                "story_id": str(story_id),
                "skill_name": skill.name,
                "rank": skill.rank,
                "description": skill.description,
                "mechanical_effect": skill.mechanical_effect,
                "requirements": skill.requirements,
                "cooldown": skill.cooldown,
            }

            if existing.data:
                self.db.table("protagonist_skills").update(skill_data).eq(
                    "id", existing.data[0]["id"]
                ).execute()
            else:
                skill_data["acquired_session_id"] = str(session_id)
                self.db.table("protagonist_skills").insert(skill_data).execute()

        # Handle inventory changes
        for inv_change in protagonist.inventory_changes:
            if inv_change.action in [
                InventoryAction.RECEIVED,
                InventoryAction.EQUIPPED,
                InventoryAction.WEARING,
            ]:
                # Add or update item
                existing = self.db.table("protagonist_inventory").select(
                    "id"
                ).eq("story_id", str(story_id)).eq(
                    "item_name", inv_change.item
                ).is_("lost_session_id", "null").execute()

                if not existing.data:
                    self.db.table("protagonist_inventory").insert({
                        "story_id": str(story_id),
                        "item_name": inv_change.item,
                        "description": inv_change.description,
                        "properties": inv_change.properties,
                        "equipped": inv_change.action in [
                            InventoryAction.EQUIPPED,
                            InventoryAction.WEARING,
                        ],
                        "acquired_session_id": str(session_id),
                    }).execute()
                elif inv_change.action in [
                    InventoryAction.EQUIPPED,
                    InventoryAction.WEARING,
                ]:
                    self.db.table("protagonist_inventory").update({
                        "equipped": True,
                    }).eq("id", existing.data[0]["id"]).execute()

            elif inv_change.action in [
                InventoryAction.LOST,
                InventoryAction.CONSUMED,
                InventoryAction.GIVEN_AWAY,
                InventoryAction.DROPPED,
            ]:
                # Mark item as lost
                self.db.table("protagonist_inventory").update({
                    "lost_session_id": str(session_id),
                }).eq("story_id", str(story_id)).eq(
                    "item_name", inv_change.item
                ).is_("lost_session_id", "null").execute()

        # Handle status effects
        for effect in protagonist.status_effects:
            existing = self.db.table("protagonist_status_effects").select(
                "id"
            ).eq("story_id", str(story_id)).eq(
                "effect_name", effect.effect
            ).is_("removed_session_id", "null").execute()

            if not existing.data:
                self.db.table("protagonist_status_effects").insert({
                    "story_id": str(story_id),
                    "effect_name": effect.effect,
                    "description": effect.description,
                    "is_temporary": effect.temporary,
                    "applied_session_id": str(session_id),
                }).execute()

    async def _update_character_state(
        self,
        story_id: UUID,
        session_id: UUID,
        char_state: CharacterState,
        entity_map: dict[str, UUID],
    ) -> None:
        """Update NPC relationship meter."""
        # Find entity
        entity_id = entity_map.get(char_state.character.lower())
        if not entity_id:
            entity = await self._find_entity_by_name_or_alias(
                story_id, char_state.character
            )
            if entity:
                entity_id = UUID(entity["id"])

        if not entity_id:
            return

        # Check for existing state
        existing = self.db.table("character_states").select("*").eq(
            "story_id", str(story_id)
        ).eq("entity_id", str(entity_id)).eq(
            "stat_type", char_state.stat_type.value
        ).execute()

        if existing.data:
            old_value = existing.data[0]["current_value"]
            state_id = existing.data[0]["id"]

            # Update state
            self.db.table("character_states").update({
                "current_value": char_state.value,
                "max_value": char_state.max,
                "label": char_state.label,
                "updated_at": datetime.utcnow().isoformat(),
            }).eq("id", state_id).execute()

            # Record history
            self.db.table("character_state_history").insert({
                "character_state_id": state_id,
                "previous_value": old_value,
                "new_value": char_state.value,
                "change_amount": char_state.value - old_value,
                "session_id": str(session_id),
            }).execute()
        else:
            # Create new state
            result = self.db.table("character_states").insert({
                "story_id": str(story_id),
                "entity_id": str(entity_id),
                "stat_type": char_state.stat_type.value,
                "current_value": char_state.value,
                "max_value": char_state.max,
                "label": char_state.label,
            }).execute()

            # Record initial value in history
            self.db.table("character_state_history").insert({
                "character_state_id": result.data[0]["id"],
                "previous_value": 0,
                "new_value": char_state.value,
                "change_amount": char_state.value,
                "session_id": str(session_id),
            }).execute()

    async def _update_session(
        self,
        session_id: UUID,
        summary: str,
        key_moments: list[str],
        raw_extraction: dict,
    ) -> None:
        """Update session with extraction results."""
        self.db.table("sessions").update({
            "summary": summary,
            "key_moments": key_moments,
            "raw_extraction": raw_extraction,
            "status": "processed",
            "processed_at": datetime.utcnow().isoformat(),
        }).eq("id", str(session_id)).execute()

    async def _update_world_state(
        self,
        story_id: UUID,
        world_state,
    ) -> None:
        """Update story's current situation from world state."""
        if world_state.current_location or world_state.current_time:
            situation_parts = []
            if world_state.current_location:
                situation_parts.append(f"Location: {world_state.current_location}")
            if world_state.current_time:
                situation_parts.append(f"Time: {world_state.current_time}")

            self.db.table("stories").update({
                "current_situation": " | ".join(situation_parts),
                "updated_at": datetime.utcnow().isoformat(),
            }).eq("id", str(story_id)).execute()

    async def get_existing_entities(
        self,
        story_id: UUID,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get existing entities for a story.

        Args:
            story_id: Story UUID
            limit: Maximum entities to return

        Returns:
            List of entity dictionaries with aliases
        """
        # Get entities ordered by importance
        result = self.db.table("entities").select(
            "id, canonical_name, entity_type, description, computed_importance"
        ).eq("story_id", str(story_id)).order(
            "computed_importance", desc=True
        ).limit(limit).execute()

        entities = []
        for entity in result.data:
            # Get aliases
            aliases_result = self.db.table("entity_aliases").select(
                "alias"
            ).eq("entity_id", entity["id"]).execute()

            aliases = [a["alias"] for a in aliases_result.data]

            entities.append({
                "id": entity["id"],
                "canonical_name": entity["canonical_name"],
                "entity_type": entity["entity_type"],
                "description": entity["description"],
                "aliases": aliases,
                "importance": entity["computed_importance"],
            })

        return entities

    async def update_story_summary(
        self,
        story_id: UUID,
        summary: str,
    ) -> None:
        """
        Update the story summary.

        Args:
            story_id: Story UUID
            summary: New summary text
        """
        self.db.table("stories").update({
            "story_summary": summary,
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("id", str(story_id)).execute()

    async def merge_entities(
        self,
        story_id: UUID,
        primary_entity_id: UUID,
        secondary_entity_id: UUID,
        reason: str,
        merged_by: str = "system",
    ) -> None:
        """
        Merge two entities when discovered to be the same.

        Moves all facts, relationships, and event participations from
        secondary to primary, then soft-deletes secondary.
        """
        # Move aliases
        self.db.table("entity_aliases").update({
            "entity_id": str(primary_entity_id),
        }).eq("entity_id", str(secondary_entity_id)).execute()

        # Move facts
        self.db.table("entity_facts").update({
            "entity_id": str(primary_entity_id),
        }).eq("entity_id", str(secondary_entity_id)).execute()

        # Move relationships (both directions)
        self.db.table("entity_relationships").update({
            "source_entity_id": str(primary_entity_id),
        }).eq("source_entity_id", str(secondary_entity_id)).execute()

        self.db.table("entity_relationships").update({
            "target_entity_id": str(primary_entity_id),
        }).eq("target_entity_id", str(secondary_entity_id)).execute()

        # Move event participations
        self.db.table("event_participants").update({
            "entity_id": str(primary_entity_id),
        }).eq("entity_id", str(secondary_entity_id)).execute()

        # Move character states
        self.db.table("character_states").update({
            "entity_id": str(primary_entity_id),
        }).eq("entity_id", str(secondary_entity_id)).execute()

        # Update mention count on primary
        secondary = self.db.table("entities").select(
            "mention_count"
        ).eq("id", str(secondary_entity_id)).execute()

        if secondary.data:
            self.db.rpc("increment_mention_count", {
                "p_entity_id": str(primary_entity_id)
            }).execute()

        # Soft delete secondary (mark as merged)
        self.db.table("entities").update({
            "status": "inactive",
            "metadata": {"merged_into": str(primary_entity_id), "merge_reason": reason},
        }).eq("id", str(secondary_entity_id)).execute()

        # Record merge in entity_merges table if it exists
        try:
            self.db.table("entity_merges").insert({
                "story_id": str(story_id),
                "primary_entity_id": str(primary_entity_id),
                "merged_entity_id": str(secondary_entity_id),
                "reason": reason,
                "merged_by": merged_by,
            }).execute()
        except Exception:
            pass  # Table might not exist yet

    async def invalidate_fact(
        self,
        fact_id: UUID,
        session_id: UUID,
        new_fact_id: Optional[UUID] = None,
    ) -> None:
        """
        Invalidate a fact (when it's discovered to be false or outdated).

        Args:
            fact_id: ID of the fact to invalidate
            session_id: Session where this was discovered
            new_fact_id: Optional ID of the fact that supersedes this one
        """
        update_data = {
            "invalidated_session_id": str(session_id),
        }
        if new_fact_id:
            update_data["superseded_by_id"] = str(new_fact_id)

        self.db.table("entity_facts").update(update_data).eq(
            "id", str(fact_id)
        ).execute()
