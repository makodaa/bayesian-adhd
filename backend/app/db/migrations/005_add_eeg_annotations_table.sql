CREATE TABLE eeg_annotations (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    clinician_id INTEGER REFERENCES clinicians(id) ON DELETE SET NULL,
    band_name VARCHAR(50) NOT NULL,
    start_time_sec FLOAT NOT NULL,
    end_time_sec FLOAT,
    label VARCHAR(100) NOT NULL,
    notes VARCHAR(255),
    color VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eeg_annotations_result ON eeg_annotations(result_id);
CREATE INDEX IF NOT EXISTS idx_eeg_annotations_result_band
    ON eeg_annotations(result_id, band_name);
