import numpy as np

APP_VERSION = "1.0"

# ============================================================================
# Signal Processing
# ============================================================================
SAMPLE_RATE = 128  # Hz
WINDOW_SECONDS = 2
OVERLAP = 0.5
SAMPLES_PER_WINDOW = int(SAMPLE_RATE * WINDOW_SECONDS)
WINDOW_STEP = int(SAMPLES_PER_WINDOW * (1 - OVERLAP))
TARGET_FREQUENCY_BINS = np.arange(2.0, 40.5, 0.5)
BROADBAND = (0.5, 40.0)
ALLOWED_EXTENSIONS = {"csv"}

# ============================================================================
# Frequency Bands (Hz)
# ============================================================================
# All available frequency bands for analysis
FREQUENCY_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 60),
    "fast_alpha": (8, 10),
    "high_beta": (15, 30),
}

# Core bands displayed in UI to end users (excludes specialized analysis bands)
DISPLAY_FREQUENCY_BANDS = {
    "delta": FREQUENCY_BANDS["delta"],
    "theta": FREQUENCY_BANDS["theta"],
    "alpha": FREQUENCY_BANDS["alpha"],
    "beta": FREQUENCY_BANDS["beta"],
    "gamma": FREQUENCY_BANDS["gamma"],
}

# Subset used by ML model for feature extraction (includes specialized bands)
MODEL_FREQUENCY_BANDS = {
    "theta": FREQUENCY_BANDS["theta"],
    "alpha": FREQUENCY_BANDS["alpha"],
    "fast_alpha": FREQUENCY_BANDS["fast_alpha"],
    "beta": FREQUENCY_BANDS["beta"],
    "high_beta": FREQUENCY_BANDS["high_beta"],
}

# ============================================================================
# Electrode Configuration (10-20 System)
# ============================================================================
ELECTRODE_CHANNELS = [
    "Fp1", "Fp2",
    "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz",
    "P3", "P4", "Pz",
    "O1", "O2",
    "T7", "T8", "P7", "P8",
]

# Old naming → New MCN naming (T3→T7, T4→T8, T5→P7, T6→P8)
OLD_TO_NEW_ELECTRODE_MAPPING = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}

# Both old and new electrode names are accepted
ALL_VALID_ELECTRODES = set(ELECTRODE_CHANNELS) | set(OLD_TO_NEW_ELECTRODE_MAPPING.keys())

# ============================================================================
# Visualization Cache
# ============================================================================
VIZ_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours
VIZ_CACHE_DETAIL_TTL_SECONDS = 24 * 60 * 60  # 24 hours
VIZ_CACHE_MAX_BYTES = 1024 * 1024 * 1024  # 1 GB
VIZ_CACHE_DIR = "/tmp/bayesian_adhd_viz_cache"

# ============================================================================
# Model
# ============================================================================
MODEL_PATH = "app/ml/optimized_model.pth"
PARAMETERS_PATH = "app/ml/best_parameters.json"
