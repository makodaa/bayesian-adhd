-- Add temporal_importance table for storing temporal sensitivity analysis results
-- This table stores time-window occlusion sensitivity analysis
-- PostgreSQL version

CREATE TABLE IF NOT EXISTS temporal_importance (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    time_points JSONB NOT NULL,  -- JSON array of time points in seconds
    importance_scores JSONB NOT NULL,  -- JSON array of importance scores
    window_size_ms INTEGER NOT NULL,  -- Size of occlusion window in milliseconds
    stride_ms INTEGER NOT NULL,  -- Stride for sliding window in milliseconds
    time_curve_plot TEXT,  -- Base64 encoded time-importance curve
    heatmap_plot TEXT,  -- Base64 encoded heatmap visualization
    statistics_plot TEXT,  -- Base64 encoded statistics plot
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_result FOREIGN KEY (result_id) REFERENCES results(id) ON DELETE CASCADE
);

-- Create index for faster lookups by result_id
CREATE INDEX IF NOT EXISTS idx_temporal_importance_result_id ON temporal_importance(result_id);

-- Create index for created_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_temporal_importance_created_at ON temporal_importance(created_at);

-- Add comment
COMMENT ON TABLE temporal_importance IS 'Stores temporal importance analysis results showing when (in time) the model is most sensitive';
