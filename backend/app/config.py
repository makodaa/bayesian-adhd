import numpy as np

# Signal processing constants

SAMPLE_RATE = 128           # Hz
WINDOW_SECONDS = 4          # Window duration
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
    'gamma': (30,60)
}

# Model paths
SCALER_PATH = 'exports/saved_scaler.pkl'
MODEL_PATH = 'exports/eeg_cnn_lstm_hpo.pth'

