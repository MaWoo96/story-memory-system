-- ============================================
-- Story Memory System - Initial Database Schema
-- ============================================
-- Run this in your Supabase SQL Editor

-- ============================================
-- CORE TABLES
-- ============================================

CREATE TABLE stories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    title TEXT NOT NULL,
    premise TEXT,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'abandoned')),
    story_summary TEXT,
    current_situation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_stories_user ON stories(user_id);

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    session_number INTEGER NOT NULL,
    transcript TEXT,
    summary TEXT,
    key_moments TEXT[],
    raw_extraction JSONB,  -- Store full extraction for debugging
    status TEXT DEFAULT 'in_progress'
        CHECK (status IN ('in_progress', 'completed', 'processing', 'processed', 'failed')),
    processed_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    UNIQUE(story_id, session_number)
);

CREATE INDEX idx_sessions_story ON sessions(story_id);

-- ============================================
-- ENTITY SYSTEM
-- ============================================

CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'active'
        CHECK (status IN ('active', 'inactive', 'deceased', 'destroyed', 'unknown')),
    first_session_id UUID REFERENCES sessions(id),
    last_session_id UUID REFERENCES sessions(id),
    mention_count INTEGER DEFAULT 1,
    base_importance REAL DEFAULT 0.5,
    computed_importance REAL DEFAULT 0.5,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(story_id, canonical_name)
);

CREATE INDEX idx_entities_story ON entities(story_id);
CREATE INDEX idx_entities_type ON entities(story_id, entity_type);
CREATE INDEX idx_entities_importance ON entities(story_id, computed_importance DESC);

CREATE TABLE entity_aliases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    alias_lower TEXT NOT NULL,
    UNIQUE(entity_id, alias_lower)
);

CREATE INDEX idx_aliases_lookup ON entity_aliases(alias_lower);

CREATE TABLE entity_facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    fact_type TEXT NOT NULL,
    fact_key TEXT,
    fact_value TEXT NOT NULL,
    established_session_id UUID REFERENCES sessions(id),
    invalidated_session_id UUID REFERENCES sessions(id),
    superseded_by_id UUID REFERENCES entity_facts(id),
    importance REAL DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_facts_entity ON entity_facts(entity_id);
CREATE INDEX idx_facts_valid ON entity_facts(entity_id, invalidated_session_id)
    WHERE invalidated_session_id IS NULL;

CREATE TABLE entity_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    source_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    description TEXT,
    established_session_id UUID REFERENCES sessions(id),
    ended_session_id UUID REFERENCES sessions(id),
    is_bidirectional BOOLEAN DEFAULT FALSE,
    importance REAL DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_relationships_source ON entity_relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON entity_relationships(target_entity_id);

-- ============================================
-- EVENTS AND DECISIONS
-- ============================================

CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    sequence_in_session INTEGER,
    importance REAL DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_events_session ON events(session_id);

CREATE TABLE event_participants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    role TEXT,
    UNIQUE(event_id, entity_id)
);

CREATE TABLE decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    situation TEXT NOT NULL,
    choice_made TEXT,
    alternatives TEXT[],
    decision_type TEXT DEFAULT 'made' CHECK (decision_type IN ('made', 'pending')),
    importance REAL DEFAULT 0.6,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_decisions_story ON decisions(story_id);

-- ============================================
-- PROTAGONIST STATE (Game Mechanics)
-- ============================================

CREATE TABLE protagonist_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    stat_name TEXT NOT NULL,
    current_value TEXT,  -- Flexible: can store numbers or status strings
    max_value TEXT,
    stat_type TEXT NOT NULL,  -- percentage, rank, integer, status
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(story_id, stat_name)
);

CREATE TABLE protagonist_skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    skill_name TEXT NOT NULL,
    rank TEXT,
    description TEXT,
    mechanical_effect TEXT,
    requirements TEXT,
    cooldown TEXT,
    acquired_session_id UUID REFERENCES sessions(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(story_id, skill_name)
);

CREATE TABLE protagonist_inventory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    item_name TEXT NOT NULL,
    description TEXT,
    properties TEXT,
    equipped BOOLEAN DEFAULT FALSE,
    acquired_session_id UUID REFERENCES sessions(id),
    lost_session_id UUID REFERENCES sessions(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_inventory_active ON protagonist_inventory(story_id, lost_session_id)
    WHERE lost_session_id IS NULL;

CREATE TABLE protagonist_status_effects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    effect_name TEXT NOT NULL,
    description TEXT,
    is_temporary BOOLEAN DEFAULT TRUE,
    applied_session_id UUID REFERENCES sessions(id),
    removed_session_id UUID REFERENCES sessions(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- NPC RELATIONSHIP METERS
-- ============================================

CREATE TABLE character_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    stat_type TEXT NOT NULL,  -- affection, trust, loyalty, fear, respect, rivalry
    current_value INTEGER NOT NULL,
    max_value INTEGER DEFAULT 100,
    label TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(story_id, entity_id, stat_type)
);

CREATE TABLE character_state_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    character_state_id UUID NOT NULL REFERENCES character_states(id) ON DELETE CASCADE,
    previous_value INTEGER,
    new_value INTEGER,
    change_amount INTEGER,
    session_id UUID REFERENCES sessions(id),
    changed_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- HIERARCHICAL SUMMARIES
-- ============================================

CREATE TABLE arc_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    arc_number INTEGER NOT NULL,
    start_session INTEGER NOT NULL,
    end_session INTEGER NOT NULL,
    summary TEXT NOT NULL,
    major_events TEXT[],
    major_decisions TEXT[],
    character_developments TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(story_id, arc_number)
);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

CREATE OR REPLACE FUNCTION increment_mention_count(p_entity_id UUID)
RETURNS void AS $$
BEGIN
    UPDATE entities
    SET mention_count = mention_count + 1,
        updated_at = NOW()
    WHERE id = p_entity_id;
END;
$$ LANGUAGE plpgsql;
