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
    password_hash VARCHAR(255),
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
    sleep_hours NUMERIC(4,2),
    food_intake TEXT,
    caffeinated BOOLEAN,
    medicated BOOLEAN,
    medication_intake TEXT,
    artifacts_noted TEXT,
    notes TEXT,
    uploaded_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE results (
    id SERIAL PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    clinician_id INTEGER REFERENCES clinicians(id) ON DELETE SET NULL,
    predicted_class VARCHAR(50) NOT NULL,
    confidence_score FLOAT NOT NULL,
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