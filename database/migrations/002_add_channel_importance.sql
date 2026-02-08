-- Add channel_importance table to store spatial feature analysis results
CREATE TABLE channel_importance (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    topographic_map TEXT NOT NULL,  -- Base64 encoded image
    bar_chart TEXT NOT NULL,  -- Base64 encoded image
    regional_chart TEXT NOT NULL,  -- Base64 encoded image
    importance_scores JSONB NOT NULL,  -- JSON object with channel: score pairs
    normalized_scores JSONB NOT NULL,  -- JSON object with channel: normalized_score pairs
    predicted_class VARCHAR(50) NOT NULL,
    confidence_score FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(result_id)  -- One channel importance per result
);

-- Index for faster lookups
CREATE INDEX idx_channel_importance_result_id ON channel_importance(result_id);
