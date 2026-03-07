-- Migration 004: Add eeg_visualizations table for persisting generated EEG images
-- These are the rendered PNG images (base64) for each frequency band + segment overlay

CREATE TABLE IF NOT EXISTS eeg_visualizations (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    band_name VARCHAR(50) NOT NULL,
    image_data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eeg_visualizations_result ON eeg_visualizations(result_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_eeg_visualizations_result_band
    ON eeg_visualizations(result_id, band_name);
