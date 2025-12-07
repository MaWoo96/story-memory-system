-- ============================================
-- Story Memory System - Image Generation Tables
-- ============================================
-- Run this after 002_add_tracking_tables.sql

-- ============================================
-- ENTITY IMAGES (Character Portraits)
-- ============================================

CREATE TABLE IF NOT EXISTS entity_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    image_type TEXT NOT NULL DEFAULT 'portrait',  -- portrait, full_body, expression, scene
    file_path TEXT NOT NULL,
    file_url TEXT,  -- Optional: if served via web server
    generation_prompt TEXT,
    negative_prompt TEXT,
    seed BIGINT,
    model_used TEXT,
    sampler TEXT,
    steps INTEGER,
    cfg_scale REAL,
    width INTEGER,
    height INTEGER,
    is_primary BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_entity_images_entity ON entity_images(entity_id);
CREATE INDEX idx_entity_images_primary ON entity_images(entity_id, is_primary) WHERE is_primary = TRUE;
CREATE INDEX idx_entity_images_type ON entity_images(entity_id, image_type);

-- ============================================
-- SCENE IMAGES
-- ============================================

CREATE TABLE IF NOT EXISTS scene_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    event_id UUID REFERENCES events(id) ON DELETE SET NULL,
    image_type TEXT NOT NULL DEFAULT 'scene',  -- scene, action, location, map
    file_path TEXT NOT NULL,
    file_url TEXT,
    generation_prompt TEXT,
    negative_prompt TEXT,
    seed BIGINT,
    model_used TEXT,
    sampler TEXT,
    steps INTEGER,
    cfg_scale REAL,
    width INTEGER,
    height INTEGER,
    participating_entities UUID[],  -- Entity IDs depicted in scene
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scene_images_story ON scene_images(story_id);
CREATE INDEX idx_scene_images_session ON scene_images(session_id);
CREATE INDEX idx_scene_images_event ON scene_images(event_id);
CREATE INDEX idx_scene_images_type ON scene_images(story_id, image_type);

-- ============================================
-- IMAGE GENERATION QUEUE (Optional - for background processing)
-- ============================================

CREATE TABLE IF NOT EXISTS image_generation_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    generation_type TEXT NOT NULL,  -- portrait, scene, variation
    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    parameters JSONB DEFAULT '{}',  -- width, height, seed, style, etc.
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    result_image_id UUID,  -- References entity_images or scene_images
    error_message TEXT,
    priority INTEGER DEFAULT 0,  -- Higher = more urgent
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_image_queue_status ON image_generation_queue(status, priority DESC);
CREATE INDEX idx_image_queue_story ON image_generation_queue(story_id);

-- ============================================
-- HELPER FUNCTION: Get primary portrait for entity
-- ============================================

CREATE OR REPLACE FUNCTION get_entity_primary_portrait(p_entity_id UUID)
RETURNS TABLE (
    file_path TEXT,
    file_url TEXT,
    seed BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT ei.file_path, ei.file_url, ei.seed
    FROM entity_images ei
    WHERE ei.entity_id = p_entity_id
    AND ei.is_primary = TRUE
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- HELPER FUNCTION: Auto-set first image as primary
-- ============================================

CREATE OR REPLACE FUNCTION set_first_image_primary()
RETURNS TRIGGER AS $$
BEGIN
    -- If this is the first image for the entity, set it as primary
    IF NOT EXISTS (
        SELECT 1 FROM entity_images
        WHERE entity_id = NEW.entity_id
        AND id != NEW.id
    ) THEN
        NEW.is_primary := TRUE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger (drop first if exists to allow re-running)
DROP TRIGGER IF EXISTS trg_set_first_image_primary ON entity_images;
CREATE TRIGGER trg_set_first_image_primary
    BEFORE INSERT ON entity_images
    FOR EACH ROW
    EXECUTE FUNCTION set_first_image_primary();

-- ============================================
-- ADD COMFYUI CONFIGURATION TABLE (Optional)
-- ============================================

CREATE TABLE IF NOT EXISTS image_generation_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id UUID REFERENCES stories(id) ON DELETE CASCADE,  -- NULL for global defaults
    config_key TEXT NOT NULL,
    config_value JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(story_id, config_key)
);

-- Insert default style configurations
INSERT INTO image_generation_config (story_id, config_key, config_value)
VALUES
    (NULL, 'default_style', '"anime"'),
    (NULL, 'default_portrait_size', '{"width": 512, "height": 768}'),
    (NULL, 'default_scene_size', '{"width": 1024, "height": 576}')
ON CONFLICT (story_id, config_key) DO NOTHING;

-- ============================================
-- ROW LEVEL SECURITY (Optional)
-- ============================================

-- Enable RLS
-- ALTER TABLE entity_images ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE scene_images ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE image_generation_queue ENABLE ROW LEVEL SECURITY;

-- Example policy (uncomment and modify based on your auth setup):
-- CREATE POLICY "Users can view images for their stories"
--     ON entity_images FOR SELECT
--     USING (entity_id IN (
--         SELECT e.id FROM entities e
--         JOIN stories s ON e.story_id = s.id
--         WHERE s.user_id = auth.uid()
--     ));
