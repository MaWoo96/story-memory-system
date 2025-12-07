-- ============================================
-- Story Memory System - Additional Tracking Tables
-- ============================================
-- Run this after 001_initial_schema.sql

-- ============================================
-- MEMORY QUERY LOGGING (for debugging and optimization)
-- ============================================

CREATE TABLE IF NOT EXISTS memory_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID REFERENCES stories(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    query_type TEXT NOT NULL,  -- 'context_build', 'search', 'entity_lookup'
    tokens_used INTEGER,
    entities_included UUID[],
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_memory_queries_story ON memory_queries(story_id);
CREATE INDEX idx_memory_queries_type ON memory_queries(query_type);
CREATE INDEX idx_memory_queries_created ON memory_queries(created_at DESC);

-- ============================================
-- ENTITY MERGE TRACKING
-- ============================================

CREATE TABLE IF NOT EXISTS entity_merges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    primary_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    merged_entity_id UUID NOT NULL,  -- Can't reference as it may be deleted
    reason TEXT,
    merged_by TEXT DEFAULT 'system',  -- 'system' or 'user'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_entity_merges_story ON entity_merges(story_id);
CREATE INDEX idx_entity_merges_primary ON entity_merges(primary_entity_id);

-- ============================================
-- ADD MISSING COLUMNS TO EXISTING TABLES
-- ============================================

-- Add tags to entities if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'entities' AND column_name = 'tags'
    ) THEN
        ALTER TABLE entities ADD COLUMN tags TEXT[] DEFAULT '{}';
        CREATE INDEX idx_entities_tags ON entities USING GIN(tags);
    END IF;
END $$;

-- Add tags to events if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'events' AND column_name = 'tags'
    ) THEN
        ALTER TABLE events ADD COLUMN tags TEXT[] DEFAULT '{}';
        CREATE INDEX idx_events_tags ON events USING GIN(tags);
    END IF;
END $$;

-- Add deleted_at to entities if not exists (soft delete support)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'entities' AND column_name = 'deleted_at'
    ) THEN
        ALTER TABLE entities ADD COLUMN deleted_at TIMESTAMPTZ;
    END IF;
END $$;

-- Add deleted_at to sessions if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sessions' AND column_name = 'deleted_at'
    ) THEN
        ALTER TABLE sessions ADD COLUMN deleted_at TIMESTAMPTZ;
    END IF;
END $$;

-- Add error_message to sessions if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sessions' AND column_name = 'error_message'
    ) THEN
        ALTER TABLE sessions ADD COLUMN error_message TEXT;
    END IF;
END $$;

-- Add token_count to sessions if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sessions' AND column_name = 'token_count'
    ) THEN
        ALTER TABLE sessions ADD COLUMN token_count INTEGER;
    END IF;
END $$;

-- Add genre and tone to stories if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stories' AND column_name = 'genre'
    ) THEN
        ALTER TABLE stories ADD COLUMN genre TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stories' AND column_name = 'tone'
    ) THEN
        ALTER TABLE stories ADD COLUMN tone TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stories' AND column_name = 'tags'
    ) THEN
        ALTER TABLE stories ADD COLUMN tags TEXT[] DEFAULT '{}';
        CREATE INDEX idx_stories_tags ON stories USING GIN(tags);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'stories' AND column_name = 'deleted_at'
    ) THEN
        ALTER TABLE stories ADD COLUMN deleted_at TIMESTAMPTZ;
    END IF;
END $$;

-- Add category to protagonist_stats if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'protagonist_stats' AND column_name = 'category'
    ) THEN
        ALTER TABLE protagonist_stats ADD COLUMN category TEXT;
    END IF;
END $$;

-- Add quantity and slot to protagonist_inventory if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'protagonist_inventory' AND column_name = 'quantity'
    ) THEN
        ALTER TABLE protagonist_inventory ADD COLUMN quantity INTEGER DEFAULT 1;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'protagonist_inventory' AND column_name = 'slot'
    ) THEN
        ALTER TABLE protagonist_inventory ADD COLUMN slot TEXT;
    END IF;
END $$;

-- Add duration to protagonist_status_effects if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'protagonist_status_effects' AND column_name = 'duration'
    ) THEN
        ALTER TABLE protagonist_status_effects ADD COLUMN duration TEXT;
    END IF;
END $$;

-- ============================================
-- HELPER FUNCTION: Update entity importance
-- ============================================

CREATE OR REPLACE FUNCTION update_entity_importance(p_story_id UUID, p_current_session INTEGER)
RETURNS void AS $$
BEGIN
    UPDATE entities e
    SET computed_importance =
        base_importance * 0.5 +
        LEAST(mention_count::real / 20, 0.3) +
        CASE
            WHEN s.session_number IS NULL THEN 0
            ELSE 0.2 * (1 - (p_current_session - s.session_number)::real / GREATEST(p_current_session, 1))
        END,
        updated_at = NOW()
    FROM sessions s
    WHERE e.story_id = p_story_id
    AND e.last_session_id = s.id
    AND e.deleted_at IS NULL;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- HELPER FUNCTION: Soft delete entity
-- ============================================

CREATE OR REPLACE FUNCTION soft_delete_entity(p_entity_id UUID)
RETURNS void AS $$
BEGIN
    UPDATE entities SET deleted_at = NOW() WHERE id = p_entity_id;
    UPDATE entity_relationships SET ended_session_id = (
        SELECT id FROM sessions
        WHERE story_id = (SELECT story_id FROM entities WHERE id = p_entity_id)
        ORDER BY session_number DESC LIMIT 1
    )
    WHERE (source_entity_id = p_entity_id OR target_entity_id = p_entity_id)
    AND ended_session_id IS NULL;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- ROW LEVEL SECURITY (Optional - uncomment if needed)
-- ============================================

-- Enable RLS on new tables
-- ALTER TABLE memory_queries ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE entity_merges ENABLE ROW LEVEL SECURITY;

-- Policies would go here based on your auth setup
