CREATE TABLE subjects (
    id SERIAL PRIMARY KEY,
    subject_code VARCHAR(50) UNIQUE NOT NULL,
    age INTEGER,
    gender VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE clinicians (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    middle_name VARCHAR(50),
    occupation VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE recordings (
    id SERIAL PRIMARY KEY,
    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    file_name VARCHAR(50) NOT NULL,
    file_path TEXT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE results (
    id SERIAL PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    predicted_class VARCHAR(50) NOT NULL,
    confidence_score FLOAT NOT NULL,
    inferenced_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE band_powers (
    id SERIAL PRIMARY KEY,
    results_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    frequency_band VARCHAR(50) NOT NULL,
    absolute_power FLOAT NOT NULL,
    relative_power FLOAT NOT NULL,
    CHECK (frequency_band IN ('delta', 'theta', 'alpha', 'beta', 'gamma'))
);

CREATE TABLE ratios (
    id SERIAL PRIMARY KEY,
    results_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    ratio_name VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL
);

CREATE TABLE reports (
    id SERIAL PRIMARY KEY,
    results_id INTEGER NOT NULL REFERENCES results(id) ON DELETE CASCADE,
    interpretation VARCHAR(50) NOT NULL,
    report_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    clinician_id INTEGER NOT NULL REFERENCES clinicians(id) ON DELETE CASCADE
);