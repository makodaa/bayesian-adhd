import io
import base64
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from ..utils.signal_processing import bandpass_df
from ..config import SAMPLE_RATE
from ..core.threader import threaded


class VisualizationService:
    """Generate EEG visualizations"""

    BAND_FILTERS = {
        'filtered': (0.5,40),
        'delta': (0.5,4),
        'theta': (4,8),
        'alpha': (8,12),
        'beta': (12,30),
        'gamma': (30,40),
        'raw': (None,None),
    }

    @threaded
    def visualize_df(self, df: pd.DataFrame, eeg_type:str) -> str:
        """Visualize Dataframe as EEG plot and return base64 image."""

        low, high = self.BAND_FILTERS.get(eeg_type, (None, None))
        
        ch_names = df.select_dtypes(include=[np.number]).columns.tolist()
        ch_types = ['eeg'] * len(ch_names)

        # Apply bandpass filter
        df = bandpass_df(df, low=low, high=high)
        df_array = df.to_numpy().T

        # Create MNE info and raw object
        info = mne.create_info(ch_names=ch_names, sfreq=SAMPLE_RATE, ch_types=ch_types)
        _, sample_count = df_array.shape
        duration = sample_count / SAMPLE_RATE

        raw = mne.io.RawArray(df_array, info)

        # Plot
        buf = io.BytesIO()
        fig = raw.plot(
            scalings='auto',
            duration=30.0,
            n_channels=len(ch_names),
            show=False,
            show_scrollbars=False,
            clipping=None,
            block=False
        )

        fig.set_size_inches(math.ceil(duration/6),6)
        fig.savefig(buf, format="png", dpi=600)
        plt.close(fig)
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"