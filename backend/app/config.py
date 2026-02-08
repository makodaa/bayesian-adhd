import numpy as np

# Signal processing constants

SAMPLE_RATE = 128  # Hz
WINDOW_SECONDS = 2  # Window duration
OVERLAP = 0.5
SAMPLES_PER_WINDOW = int(SAMPLE_RATE * WINDOW_SECONDS)
WINDOW_STEP = int(SAMPLES_PER_WINDOW * (1 - OVERLAP))  # step size for sliding window
TARGET_FREQUENCY_BINS = np.arange(2.0, 40.5, 0.5)  # target frequency bins
ALLOWED_EXTENSIONS = {"csv"}

# Frequency Bands & Ranges (in hz)
CLASSIFYING_FREQUENCY_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 60),
    "fast_alpha": (8, 10),
    "high_beta": (15, 30),
}

DISPLAY_FREQUENCY_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 60),
}


ELECTRODE_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
]

# Standard 10-20 EEG electrode positions (normalized x, y coordinates)
# For topographic visualization and spatial feature mapping
# Coordinates are in a unit circle (approx -1 to 1 for both x and y)
# x: left (-) to right (+), y: back (-) to front (+)
ELECTRODE_POSITIONS = {
    "Fp1": (-0.31, 0.95),  # Left frontal pole
    "Fp2": (0.31, 0.95),   # Right frontal pole
    "F7": (-0.75, 0.55),   # Left frontal lateral
    "F3": (-0.45, 0.60),   # Left frontal
    "Fz": (0.0, 0.65),     # Frontal midline
    "F4": (0.45, 0.60),    # Right frontal
    "F8": (0.75, 0.55),    # Right frontal lateral
    "T7": (-0.95, 0.0),    # Left temporal
    "C3": (-0.55, 0.0),    # Left central
    "Cz": (0.0, 0.0),      # Central midline
    "C4": (0.55, 0.0),     # Right central
    "T8": (0.95, 0.0),     # Right temporal
    "P7": (-0.75, -0.55),  # Left parietal lateral
    "P3": (-0.45, -0.60),  # Left parietal
    "Pz": (0.0, -0.65),    # Parietal midline
    "P4": (0.45, -0.60),   # Right parietal
    "P8": (0.75, -0.55),   # Right parietal lateral
    "O1": (-0.31, -0.95),  # Left occipital
    "O2": (0.31, -0.95),   # Right occipital
}

# Model paths
MODEL_PATH = "app/ml/optimized_model.pth"
