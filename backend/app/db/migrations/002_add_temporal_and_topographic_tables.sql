-- Migration: Store temporal biomarker plots and topographic maps in the database
-- so they can be loaded from stored results without re-uploading the EEG file.

-- Temporal biomarker plots (one row per group plot per result)
CREATE TABLE IF NOT EXISTS temporal_plots (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    group_name VARCHAR(100) NOT NULL,
    plot_image TEXT NOT NULL,  -- base64-encoded PNG data URI
    created_at TIMESTAMP DEFAULT NOW()
);

-- Temporal biomarker summary statistics (one row per biomarker per result)
CREATE TABLE IF NOT EXISTS temporal_summaries (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    biomarker_key VARCHAR(100) NOT NULL,
    mean_value FLOAT NOT NULL,
    std_value FLOAT NOT NULL,
    min_value FLOAT NOT NULL,
    max_value FLOAT NOT NULL
);

-- Topographic map images (one row per map per result)
CREATE TABLE IF NOT EXISTS topographic_maps (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    map_type VARCHAR(20) NOT NULL,  -- 'absolute', 'relative', 'tbr'
    band VARCHAR(20),               -- 'delta','theta','alpha','beta','gamma' or NULL for tbr
    map_image TEXT NOT NULL,         -- base64-encoded PNG data URI
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for fast lookups by result_id
CREATE INDEX IF NOT EXISTS idx_temporal_plots_result ON temporal_plots(result_id);
CREATE INDEX IF NOT EXISTS idx_temporal_summaries_result ON temporal_summaries(result_id);
CREATE INDEX IF NOT EXISTS idx_topographic_maps_result ON topographic_maps(result_id);
