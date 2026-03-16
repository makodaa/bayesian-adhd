CREATE TABLE subjects (
    id SERIAL PRIMARY KEY,
    subject_code VARCHAR(50) UNIQUE NOT NULL,
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
    created_at TIMESTAMP DEFAULT NOW()
);

-- Track active clinician sessions
CREATE TABLE clinician_sessions (
    clinician_id INTEGER PRIMARY KEY REFERENCES clinicians(id) ON DELETE CASCADE,
    logged_in_at TIMESTAMP DEFAULT NOW()
);

-- Insert Admin Clinician with password adminclinician123
INSERT INTO clinicians (first_name, last_name, middle_name, occupation, password_hash)
VALUES ('Admin', 'Clinician', '', 'Administrator', 'scrypt:32768:8:1$W2H7WyHgIQs6dzbI$25d334197247946f01845eb3f8a041e9711b8b2be04c3cd3876c228fbe1bbfa718ef843c7dfd49f1fabcbd0e2c7dbd339a4c1a7c120202c094c5a01aa439fecb');

CREATE TABLE recordings (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(50) NOT NULL,
    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    technician_name TEXT,
    sleep_hours NUMERIC(4,2) CHECK (sleep_hours >= 0),
    coffee_hours_ago NUMERIC(4,2) NOT NULL CHECK (coffee_hours_ago >= 0),
    drugs_hours_ago NUMERIC(4,2) NOT NULL CHECK (drugs_hours_ago >= 0),
    meal_hours_ago NUMERIC(4,2) NOT NULL CHECK (meal_hours_ago >= 0),
    medication TEXT,
    recorded_minutes NUMERIC(6,2),
    duration_minutes NUMERIC(6,2),
    artifacts_noted TEXT,
    notes TEXT,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE results (
    id SERIAL PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    clinician_id INTEGER REFERENCES clinicians(id) ON DELETE SET NULL,
    predicted_class VARCHAR(50) NOT NULL,
    confidence_score FLOAT NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
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

CREATE INDEX IF NOT EXISTS idx_temporal_plots_result ON temporal_plots(result_id);
CREATE INDEX IF NOT EXISTS idx_temporal_summaries_result ON temporal_summaries(result_id);
CREATE INDEX IF NOT EXISTS idx_topographic_maps_result ON topographic_maps(result_id);
CREATE INDEX IF NOT EXISTS idx_eeg_visualizations_result ON eeg_visualizations(result_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_eeg_visualizations_result_band
    ON eeg_visualizations(result_id, band_name);
