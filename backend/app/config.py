import numpy as np

# Signal processing constants

SAMPLE_RATE = 128           # Hz
WINDOW_SECONDS = 2          # Window duration
OVERLAP = 0.5
SAMPLES_PER_WINDOW = int (SAMPLE_RATE * WINDOW_SECONDS)
WINDOW_STEP = int(SAMPLES_PER_WINDOW * (1-OVERLAP))     # step size for sliding window
TARGET_FREQUENCY_BINS = np.arange(2.0,40.5,0.5)        # target frequency bins
ALLOWED_EXTENSIONS = {'csv'}

# Frequency Bands & Ranges (in hz)
FREQUENCY_BANDS = {
    'delta': (0.5,4),
    'theta': (4,8),
    'alpha': (8,13),
    'beta': (13,30),
    'gamma': (30,60),
    'fast_alpha': (8, 10),
    'high_beta': (15, 30),
}


ELECTRODE_CHANNELS = ["Fp1","Fp2","F3","F4","C3",
                      "C4","P3","P4","O1","O2",
                      "F7","F8","T7","T8","P7",
                      "P8","Fz","Cz","Pz"]

# Model paths
MODEL_PATH = 'app/ml/optimized_model.pth'

