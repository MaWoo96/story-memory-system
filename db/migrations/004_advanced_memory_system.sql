-- ============================================
-- Story Memory System - Advanced Memory & State Tracking
-- ============================================
-- Run this after 003_add_image_tables.sql
-- Adds: Physical states, intimacy metrics, scene tracking,
-- semantic memories, hierarchical summaries, user-editable memories

-- ============================================
-- PHYSICAL STATE TRACKING
-- ============================================

CREATE TABLE IF NOT EXISTS character_physical_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    character_name TEXT NOT NULL,  -- 'protagonist' or entity name
    entity_id UUID REFERENCES entities(id) ON DELETE SET NULL,  -- Optional link to entity

    -- Physical state components
    clothing JSONB DEFAULT '[]',  -- Array of clothing items
    position TEXT,  -- standing, sitting, lying, kneeling, etc.
    location_in_scene TEXT,  -- on bed, by window, etc.
    physical_contact JSONB DEFAULT '[]',  -- Array of contact descriptions
    temporary_states JSONB DEFAULT '[]',  -- arousal, exhaustion, etc.

    -- Metadata
    is_current BOOLEAN DEFAULT TRUE,  -- Is this the latest state?
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_physical_states_story ON character_physical_states(story_id);
CREATE INDEX idx_physical_states_session ON character_physical_states(session_id);
CREATE INDEX idx_physical_states_current ON character_physical_states(story_id, is_current)
    WHERE is_current = TRUE;

-- ============================================
-- INTIMACY METRICS (Multi-dimensional relationships)
-- ============================================

CREATE TABLE IF NOT EXISTS intimacy_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,

    -- Core metrics (0-100)
    affection INTEGER DEFAULT 50 CHECK (affection >= 0 AND affection <= 100),
    trust INTEGER DEFAULT 50 CHECK (trust >= 0 AND trust <= 100),
    lust INTEGER DEFAULT 0 CHECK (lust >= 0 AND lust <= 100),
    comfort INTEGER DEFAULT 50 CHECK (comfort >= 0 AND comfort <= 100),
    jealousy INTEGER DEFAULT 0 CHECK (jealousy >= 0 AND jealousy <= 100),

    -- Optional dynamics (null if not relevant)
    submission INTEGER CHECK (submission IS NULL OR (submission >= 0 AND submission <= 100)),
    dominance INTEGER CHECK (dominance IS NULL OR (dominance >= 0 AND dominance <= 100)),

    -- Metadata
    last_session_id UUID REFERENCES sessions(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(story_id, entity_id)
);

CREATE INDEX idx_intimacy_metrics_story ON intimacy_metrics(story_id);
CREATE INDEX idx_intimacy_metrics_entity ON intimacy_metrics(entity_id);

-- History of intimacy changes
CREATE TABLE IF NOT EXISTS intimacy_metrics_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    intimacy_metric_id UUID NOT NULL REFERENCES intimacy_metrics(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id),

    -- What changed
    metric_name TEXT NOT NULL,  -- affection, trust, lust, etc.
    old_value INTEGER,
    new_value INTEGER,
    change_amount INTEGER,
    reason TEXT,  -- Why it changed

    changed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_intimacy_history_metric ON intimacy_metrics_history(intimacy_metric_id);
CREATE INDEX idx_intimacy_history_session ON intimacy_metrics_history(session_id);

-- ============================================
-- SCENE STATE TRACKING
-- ============================================

CREATE TABLE IF NOT EXISTS scene_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,

    -- Scene info
    scene_type TEXT NOT NULL,  -- dialogue, action, intimate, combat, etc.
    scene_active BOOLEAN DEFAULT TRUE,
    participants JSONB DEFAULT '[]',  -- Array of character names
    mood TEXT,
    interrupted BOOLEAN DEFAULT FALSE,
    consent_established JSONB DEFAULT '[]',  -- For intimate scenes

    -- Tracking
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    sequence_number INTEGER,  -- Order within session

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scene_states_story ON scene_states(story_id);
CREATE INDEX idx_scene_states_session ON scene_states(session_id);
CREATE INDEX idx_scene_states_active ON scene_states(story_id, scene_active)
    WHERE scene_active = TRUE;

-- ============================================
-- SEMANTIC MEMORIES (SpicyChat pattern)
-- ============================================

CREATE TABLE IF NOT EXISTS semantic_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,

    -- Memory content
    memory_text TEXT NOT NULL,  -- Compressed 1-2 sentence memory
    characters_involved JSONB DEFAULT '[]',  -- Array of character names
    primary_emotion TEXT NOT NULL,  -- joy, tension, intimacy, conflict, etc.
    topics JSONB DEFAULT '[]',  -- Array of topics

    -- Importance and retrieval
    importance REAL DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,

    -- Setup/Payoff tracking
    setup_for_payoff BOOLEAN DEFAULT FALSE,
    payoff_for_id UUID REFERENCES semantic_memories(id),

    -- User control (editable memories)
    pinned BOOLEAN DEFAULT FALSE,  -- Always include in context
    user_edited BOOLEAN DEFAULT FALSE,
    hidden BOOLEAN DEFAULT FALSE,  -- User chose to hide

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_semantic_memories_story ON semantic_memories(story_id);
CREATE INDEX idx_semantic_memories_session ON semantic_memories(session_id);
CREATE INDEX idx_semantic_memories_importance ON semantic_memories(story_id, importance DESC);
CREATE INDEX idx_semantic_memories_pinned ON semantic_memories(story_id, pinned) WHERE pinned = TRUE;
CREATE INDEX idx_semantic_memories_emotion ON semantic_memories(story_id, primary_emotion);
CREATE INDEX idx_semantic_memories_topics ON semantic_memories USING GIN(topics);

-- ============================================
-- HIERARCHICAL SUMMARIES
-- ============================================

-- Scene-level summaries (every ~30 messages)
CREATE TABLE IF NOT EXISTS scene_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,

    scene_number INTEGER NOT NULL,
    summary TEXT NOT NULL,
    characters_present JSONB DEFAULT '[]',
    key_events JSONB DEFAULT '[]',
    mood TEXT,

    message_start INTEGER,  -- First message index
    message_end INTEGER,    -- Last message index

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scene_summaries_story ON scene_summaries(story_id);
CREATE INDEX idx_scene_summaries_session ON scene_summaries(session_id);

-- Chapter-level summaries (every ~5 scenes or session boundary)
CREATE TABLE IF NOT EXISTS chapter_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,

    chapter_number INTEGER NOT NULL,
    title TEXT,
    summary TEXT NOT NULL,
    major_events JSONB DEFAULT '[]',
    character_developments JSONB DEFAULT '[]',
    relationship_changes JSONB DEFAULT '[]',

    start_session INTEGER NOT NULL,
    end_session INTEGER NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(story_id, chapter_number)
);

CREATE INDEX idx_chapter_summaries_story ON chapter_summaries(story_id);

-- Story arc summaries (highest level)
-- Already exists in 001_initial_schema.sql as arc_summaries

-- ============================================
-- ADD STORY-LEVEL SETTINGS
-- ============================================

-- Add columns to stories table for new features
DO $$
BEGIN
    -- NSFW settings
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stories' AND column_name = 'is_nsfw'
    ) THEN
        ALTER TABLE stories ADD COLUMN is_nsfw BOOLEAN DEFAULT TRUE;
    END IF;

    -- Grok instructions
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stories' AND column_name = 'grok_instructions'
    ) THEN
        ALTER TABLE stories ADD COLUMN grok_instructions TEXT;
    END IF;

    -- World setting (JSONB from wizard)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stories' AND column_name = 'world_setting'
    ) THEN
        ALTER TABLE stories ADD COLUMN world_setting JSONB;
    END IF;

    -- Description field
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stories' AND column_name = 'description'
    ) THEN
        ALTER TABLE stories ADD COLUMN description TEXT;
    END IF;

    -- Cover image URL
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stories' AND column_name = 'cover_image_url'
    ) THEN
        ALTER TABLE stories ADD COLUMN cover_image_url TEXT;
    END IF;
END $$;

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to update physical state (marks old as not current)
CREATE OR REPLACE FUNCTION update_physical_state(
    p_story_id UUID,
    p_character_name TEXT,
    p_session_id UUID,
    p_clothing JSONB,
    p_position TEXT,
    p_location_in_scene TEXT,
    p_physical_contact JSONB,
    p_temporary_states JSONB
)
RETURNS UUID AS $$
DECLARE
    v_new_id UUID;
BEGIN
    -- Mark existing states as not current
    UPDATE character_physical_states
    SET is_current = FALSE, updated_at = NOW()
    WHERE story_id = p_story_id
    AND character_name = p_character_name
    AND is_current = TRUE;

    -- Insert new state
    INSERT INTO character_physical_states (
        story_id, session_id, character_name, clothing, position,
        location_in_scene, physical_contact, temporary_states, is_current
    ) VALUES (
        p_story_id, p_session_id, p_character_name, p_clothing, p_position,
        p_location_in_scene, p_physical_contact, p_temporary_states, TRUE
    ) RETURNING id INTO v_new_id;

    RETURN v_new_id;
END;
$$ LANGUAGE plpgsql;

-- Function to update intimacy metric with history
CREATE OR REPLACE FUNCTION update_intimacy_metric(
    p_story_id UUID,
    p_entity_id UUID,
    p_metric_name TEXT,
    p_new_value INTEGER,
    p_session_id UUID,
    p_reason TEXT DEFAULT NULL
)
RETURNS void AS $$
DECLARE
    v_intimacy_id UUID;
    v_old_value INTEGER;
BEGIN
    -- Get or create intimacy metrics record
    SELECT id INTO v_intimacy_id FROM intimacy_metrics
    WHERE story_id = p_story_id AND entity_id = p_entity_id;

    IF v_intimacy_id IS NULL THEN
        INSERT INTO intimacy_metrics (story_id, entity_id, last_session_id)
        VALUES (p_story_id, p_entity_id, p_session_id)
        RETURNING id INTO v_intimacy_id;
    END IF;

    -- Get old value
    EXECUTE format('SELECT %I FROM intimacy_metrics WHERE id = $1', p_metric_name)
    INTO v_old_value USING v_intimacy_id;

    -- Update the metric
    EXECUTE format('UPDATE intimacy_metrics SET %I = $1, last_session_id = $2, updated_at = NOW() WHERE id = $3', p_metric_name)
    USING p_new_value, p_session_id, v_intimacy_id;

    -- Record history
    INSERT INTO intimacy_metrics_history (
        intimacy_metric_id, session_id, metric_name, old_value, new_value, change_amount, reason
    ) VALUES (
        v_intimacy_id, p_session_id, p_metric_name, v_old_value, p_new_value,
        p_new_value - COALESCE(v_old_value, 50), p_reason
    );
END;
$$ LANGUAGE plpgsql;

-- Function to get memory retrieval score
CREATE OR REPLACE FUNCTION get_memory_retrieval_score(
    p_memory_id UUID,
    p_query_emotion TEXT DEFAULT NULL,
    p_query_characters TEXT[] DEFAULT NULL,
    p_hours_since_creation REAL DEFAULT 0
)
RETURNS REAL AS $$
DECLARE
    v_memory semantic_memories%ROWTYPE;
    v_score REAL;
    v_recency_score REAL;
    v_emotion_score REAL;
    v_character_overlap REAL;
BEGIN
    SELECT * INTO v_memory FROM semantic_memories WHERE id = p_memory_id;

    IF v_memory IS NULL THEN
        RETURN 0;
    END IF;

    -- Recency score (decay factor 0.995 per hour)
    v_recency_score := POWER(0.995, p_hours_since_creation);

    -- Emotion match score
    IF p_query_emotion IS NOT NULL AND v_memory.primary_emotion = p_query_emotion THEN
        v_emotion_score := 1.0;
    ELSE
        v_emotion_score := 0.0;
    END IF;

    -- Character overlap score
    IF p_query_characters IS NOT NULL AND array_length(p_query_characters, 1) > 0 THEN
        SELECT COALESCE(
            array_length(
                ARRAY(SELECT jsonb_array_elements_text(v_memory.characters_involved))
                & p_query_characters, 1
            )::REAL / GREATEST(array_length(p_query_characters, 1), 1), 0
        ) INTO v_character_overlap;
    ELSE
        v_character_overlap := 0.5;
    END IF;

    -- Combined score (weights from research)
    v_score :=
        0.10 * v_recency_score +
        0.20 * v_emotion_score +
        0.25 * v_character_overlap +
        0.15 * v_memory.importance +
        0.10 * LEAST(v_memory.access_count::REAL / 10, 1.0) +
        0.20;  -- Base semantic similarity placeholder (needs vector search)

    -- Boost pinned memories
    IF v_memory.pinned THEN
        v_score := v_score + 0.3;
    END IF;

    RETURN LEAST(v_score, 1.0);
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- VIEWS FOR COMMON QUERIES
-- ============================================

-- Current physical states for a story
CREATE OR REPLACE VIEW current_physical_states AS
SELECT * FROM character_physical_states WHERE is_current = TRUE;

-- Active scenes
CREATE OR REPLACE VIEW active_scenes AS
SELECT * FROM scene_states WHERE scene_active = TRUE;

-- Top memories for a story (by importance)
CREATE OR REPLACE VIEW top_memories AS
SELECT
    sm.*,
    e.canonical_name as first_character
FROM semantic_memories sm
LEFT JOIN LATERAL (
    SELECT canonical_name FROM entities
    WHERE canonical_name = ANY(ARRAY(SELECT jsonb_array_elements_text(sm.characters_involved)))
    LIMIT 1
) e ON true
WHERE sm.hidden = FALSE
ORDER BY sm.pinned DESC, sm.importance DESC, sm.created_at DESC;
