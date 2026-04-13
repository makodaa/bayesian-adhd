CREATE TABLE subjects (
    id SERIAL PRIMARY KEY,
    subject_code VARCHAR(60) UNIQUE NOT NULL,
    age INTEGER CHECK (age >= 1 AND age <= 120),
    date_of_birth DATE,
    gender VARCHAR(20) CHECK (gender IN ('Male', 'Female', 'Other', 'Prefer not to say')),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE clinicians (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    middle_name VARCHAR(50),
    occupation VARCHAR(50) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    archived_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Track active clinician sessions
CREATE TABLE clinician_sessions (
    clinician_id INTEGER PRIMARY KEY REFERENCES clinicians(id) ON DELETE CASCADE,
    logged_in_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE clinician_band_thresholds (
    id SERIAL PRIMARY KEY,
    clinician_id INTEGER NOT NULL REFERENCES clinicians(id) ON DELETE CASCADE,
    adhd_subtype VARCHAR(40) NOT NULL,
    band VARCHAR(20) NOT NULL,
    min_value FLOAT NOT NULL,
    max_value FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CHECK (adhd_subtype IN ('inattentive', 'hyperactive_impulsive', 'combined')),
    CHECK (band IN ('delta', 'theta', 'alpha', 'beta', 'gamma')),
    CHECK (min_value >= 0.0 AND max_value <= 1.0 AND min_value <= max_value)
);

CREATE TABLE clinician_recommendations (
    id SERIAL PRIMARY KEY,
    clinician_id INTEGER NOT NULL REFERENCES clinicians(id) ON DELETE CASCADE,
    adhd_subtype VARCHAR(40) NOT NULL,
    band VARCHAR(20) NOT NULL,
    band_state VARCHAR(20) NOT NULL,
    recommendation_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CHECK (adhd_subtype IN ('inattentive', 'hyperactive_impulsive', 'combined')),
    CHECK (band IN ('alpha', 'beta', 'theta')),
    CHECK (band_state IN ('decreased', 'elevated'))
);

CREATE TABLE clinician_subtype_recommendations (
    id SERIAL PRIMARY KEY,
    clinician_id INTEGER NOT NULL REFERENCES clinicians(id) ON DELETE CASCADE,
    adhd_subtype VARCHAR(40) NOT NULL,
    trigger_key VARCHAR(40) NOT NULL,
    recommendation_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CHECK (adhd_subtype IN ('inattentive', 'hyperactive_impulsive', 'combined')),
    CHECK (trigger_key IN ('consistent_subtype'))
);

-- Insert Admin Clinician with password adminclinician123
INSERT INTO clinicians (first_name, last_name, middle_name, occupation, password_hash)
VALUES ('Admin', 'Clinician', '', 'Administrator', 'scrypt:32768:8:1$W2H7WyHgIQs6dzbI$25d334197247946f01845eb3f8a041e9711b8b2be04c3cd3876c228fbe1bbfa718ef843c7dfd49f1fabcbd0e2c7dbd339a4c1a7c120202c094c5a01aa439fecb');

-- Seed default ranges and interventions for the admin clinician
WITH admin AS (
    SELECT id
    FROM clinicians
    WHERE first_name = 'Admin' AND last_name = 'Clinician'
    ORDER BY id
    LIMIT 1
)
INSERT INTO clinician_band_thresholds
    (clinician_id, adhd_subtype, band, min_value, max_value, created_at, updated_at)
SELECT
    admin.id,
    values_data.adhd_subtype,
    values_data.band,
    values_data.min_value,
    values_data.max_value,
    NOW(),
    NOW()
FROM admin
JOIN (
    VALUES
        ('inattentive', 'delta', 0.40, 0.90),
        ('inattentive', 'theta', 0.10, 0.25),
        ('inattentive', 'alpha', 0.03, 0.15),
        ('inattentive', 'beta', 0.02, 0.15),
        ('inattentive', 'gamma', 0.00, 0.08),
        ('hyperactive_impulsive', 'delta', 0.40, 0.90),
        ('hyperactive_impulsive', 'theta', 0.05, 0.20),
        ('hyperactive_impulsive', 'alpha', 0.06, 0.18),
        ('hyperactive_impulsive', 'beta', 0.05, 0.18),
        ('hyperactive_impulsive', 'gamma', 0.00, 0.08),
        ('combined', 'delta', 0.40, 0.90),
        ('combined', 'theta', 0.05, 0.20),
        ('combined', 'alpha', 0.03, 0.15),
        ('combined', 'beta', 0.02, 0.15),
        ('combined', 'gamma', 0.00, 0.08)
) AS values_data(adhd_subtype, band, min_value, max_value)
    ON TRUE
WHERE NOT EXISTS (
    SELECT 1
    FROM clinician_band_thresholds existing
    WHERE existing.clinician_id = admin.id
      AND existing.adhd_subtype = values_data.adhd_subtype
      AND existing.band = values_data.band
);

WITH admin AS (
    SELECT id
    FROM clinicians
    WHERE first_name = 'Admin' AND last_name = 'Clinician'
    ORDER BY id
    LIMIT 1
)
INSERT INTO clinician_recommendations
    (clinician_id, adhd_subtype, band, band_state, recommendation_text, created_at, updated_at)
SELECT
    admin.id,
    values_data.adhd_subtype,
    values_data.band,
    values_data.band_state,
    values_data.recommendation_text,
    NOW(),
    NOW()
FROM admin
JOIN (
    VALUES
        ('inattentive', 'alpha', 'decreased', 'Support focus with structured routines and shorter tasks.'),
        ('inattentive', 'alpha', 'elevated', 'Encourage active engagement and break long tasks into segments.'),
        ('inattentive', 'beta', 'decreased', 'Promote alertness with brief movement and clear task cues.'),
        ('inattentive', 'beta', 'elevated', 'Use calming pacing and reduce competing stimuli.'),
        ('inattentive', 'theta', 'decreased', 'Monitor fatigue and allow regular rest breaks.'),
        ('inattentive', 'theta', 'elevated', 'Use attention supports and minimize distractions.'),
        ('hyperactive_impulsive', 'alpha', 'decreased', 'Add calming strategies and reduce sensory load.'),
        ('hyperactive_impulsive', 'alpha', 'elevated', 'Channel energy into planned movement and short tasks.'),
        ('hyperactive_impulsive', 'beta', 'decreased', 'Use clear structure and frequent check-ins.'),
        ('hyperactive_impulsive', 'beta', 'elevated', 'Incorporate relaxation techniques and slow pacing.'),
        ('hyperactive_impulsive', 'theta', 'decreased', 'Watch for overstimulation and offer quiet breaks.'),
        ('hyperactive_impulsive', 'theta', 'elevated', 'Reinforce self-regulation and guided breathing.'),
        ('combined', 'alpha', 'decreased', 'Balance focus supports with calming routines.'),
        ('combined', 'alpha', 'elevated', 'Use engaging tasks with predictable structure.'),
        ('combined', 'beta', 'decreased', 'Provide clear prompts and short task cycles.'),
        ('combined', 'beta', 'elevated', 'Reduce pacing demands and reinforce relaxation.'),
        ('combined', 'theta', 'decreased', 'Allow recovery time and monitor fatigue.'),
        ('combined', 'theta', 'elevated', 'Use attention aids and simplify task environment.')
) AS values_data(adhd_subtype, band, band_state, recommendation_text)
    ON TRUE
WHERE NOT EXISTS (
    SELECT 1
    FROM clinician_recommendations existing
    WHERE existing.clinician_id = admin.id
      AND existing.adhd_subtype = values_data.adhd_subtype
      AND existing.band = values_data.band
      AND existing.band_state = values_data.band_state
);

WITH admin AS (
    SELECT id
    FROM clinicians
    WHERE first_name = 'Admin' AND last_name = 'Clinician'
    ORDER BY id
    LIMIT 1
)
INSERT INTO clinician_subtype_recommendations
    (clinician_id, adhd_subtype, trigger_key, recommendation_text, created_at, updated_at)
SELECT
    admin.id,
    values_data.adhd_subtype,
    values_data.trigger_key,
    values_data.recommendation_text,
    NOW(),
    NOW()
FROM admin
JOIN (
    VALUES
        ('inattentive', 'consistent_subtype', 'Pattern suggests inattentive presentation; prioritize focus supports.'),
        ('hyperactive_impulsive', 'consistent_subtype', 'Pattern suggests hyperactive-impulsive presentation; emphasize regulation strategies.'),
        ('combined', 'consistent_subtype', 'Pattern suggests combined presentation; balance focus and regulation supports.')
) AS values_data(adhd_subtype, trigger_key, recommendation_text)
    ON TRUE
WHERE NOT EXISTS (
    SELECT 1
    FROM clinician_subtype_recommendations existing
    WHERE existing.clinician_id = admin.id
      AND existing.adhd_subtype = values_data.adhd_subtype
      AND existing.trigger_key = values_data.trigger_key
);

CREATE TABLE recordings (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(50) NOT NULL,
    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    referral_name VARCHAR(60),
    referral_institution VARCHAR(60),
    technician_name VARCHAR(60),
    sleep_hours NUMERIC(4,2) NOT NULL CHECK (sleep_hours >= 0),
    coffee_hours_ago NUMERIC(4,2) CHECK (coffee_hours_ago >= 0),
    drugs_hours_ago NUMERIC(4,2) CHECK (drugs_hours_ago >= 0),
    meal_hours_ago NUMERIC(4,2) NOT NULL CHECK (meal_hours_ago >= 0),
    medication VARCHAR(255),
    recorded_minutes NUMERIC(6,2),
    duration_minutes NUMERIC(6,2),
    artifacts_noted TEXT,
    notes VARCHAR(255),
    uploaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE results (
    id SERIAL PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    clinician_id INTEGER REFERENCES clinicians(id) ON DELETE SET NULL,
    predicted_class VARCHAR(50) NOT NULL,
    confidence_score FLOAT NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    preprocessing_summary JSONB,
    inferenced_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE band_powers (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    electrode VARCHAR(50) NOT NULL,
    frequency_band VARCHAR(50) NOT NULL,
    absolute_power FLOAT NOT NULL,
    relative_power FLOAT NOT NULL,
    CHECK (frequency_band IN ('delta', 'theta', 'alpha', 'beta', 'gamma', 'fast_alpha', 'high_beta'))
);

CREATE TABLE ratios (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    ratio_name VARCHAR(50) NOT NULL,
    ratio_value FLOAT NOT NULL
);

CREATE TABLE reports (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    interpretation VARCHAR(50) NOT NULL,
    report_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    clinician_id INTEGER NOT NULL REFERENCES clinicians(id) ON DELETE CASCADE
);

CREATE TABLE temporal_plots (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    group_name VARCHAR(100) NOT NULL,
    plot_image TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE temporal_summaries (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    biomarker_key VARCHAR(100) NOT NULL,
    mean_value FLOAT NOT NULL,
    std_value FLOAT NOT NULL,
    min_value FLOAT NOT NULL,
    max_value FLOAT NOT NULL
);

CREATE TABLE topographic_maps (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    map_type VARCHAR(20) NOT NULL,
    band VARCHAR(20),
    map_image TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE eeg_visualizations (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    band_name VARCHAR(50) NOT NULL,
    image_data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE eeg_annotations (
    id SERIAL PRIMARY KEY,
    result_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    clinician_id INTEGER REFERENCES clinicians(id) ON DELETE SET NULL,
    band_name VARCHAR(50) NOT NULL,
    start_time_sec FLOAT NOT NULL,
    end_time_sec FLOAT,
    lane_start FLOAT,
    lane_end FLOAT,
    label VARCHAR(100) NOT NULL,
    notes VARCHAR(255),
    color VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_temporal_plots_result ON temporal_plots(result_id);
CREATE INDEX IF NOT EXISTS idx_temporal_summaries_result ON temporal_summaries(result_id);
CREATE INDEX IF NOT EXISTS idx_topographic_maps_result ON topographic_maps(result_id);
CREATE INDEX IF NOT EXISTS idx_eeg_visualizations_result ON eeg_visualizations(result_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_eeg_visualizations_result_band
    ON eeg_visualizations(result_id, band_name);
CREATE INDEX IF NOT EXISTS idx_eeg_annotations_result ON eeg_annotations(result_id);
CREATE INDEX IF NOT EXISTS idx_eeg_annotations_result_band
    ON eeg_annotations(result_id, band_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_clinician_band_thresholds_unique
    ON clinician_band_thresholds(clinician_id, adhd_subtype, band);
CREATE INDEX IF NOT EXISTS idx_clinician_band_thresholds_clinician
    ON clinician_band_thresholds(clinician_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_clinician_recommendations_unique
    ON clinician_recommendations(clinician_id, adhd_subtype, band, band_state);
CREATE INDEX IF NOT EXISTS idx_clinician_recommendations_clinician
    ON clinician_recommendations(clinician_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_clinician_subtype_recommendations_unique
    ON clinician_subtype_recommendations(clinician_id, adhd_subtype, trigger_key);
CREATE INDEX IF NOT EXISTS idx_clinician_subtype_recommendations_clinician
    ON clinician_subtype_recommendations(clinician_id);
