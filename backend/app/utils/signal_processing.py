import pandas as pd
import numpy as np
from ..config import SAMPLE_RATE
from scipy.signal import butter, filtfilt

def bandpass_df(df:pd.DataFrame, low=None, high=None, order=4, window_sec=1):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_samples = len(df)
    window_size = int(window_sec * SAMPLE_RATE)

    nyquist = 0.5 * SAMPLE_RATE # Nyquist frequency (realistic frequency)
    if low is None and high is None:
        return df
    
    # Normalized cutoffs
    low_cut = low / nyquist if low else None
    high_cut = high / nyquist if high else None

    # Filter design
    if low_cut and high_cut:
        btype, Wn = 'band', [low_cut, high_cut]
    elif low_cut:
        btype, Wn = 'highpass', low_cut
    elif high_cut:
        btype, Wn = 'lowpass', high_cut

    b,a = butter(order, Wn, btype=btype)

    # Apply filter in 1-second windows
    for col in numeric_cols:
        signal = df[col].to_numpy()
        filtered = np.zeros_like(signal)

        for start in range(0, n_samples, window_size):
            end = min(start +  window_size, n_samples)
            segment = signal[start:end]

            if len(segment) < order * 3:
                filtered[start:end] = segment
                continue

            filtered[start:end] = filtfilt(b,a,segment,method="gust")
    
    df[col] = filtered

    return df