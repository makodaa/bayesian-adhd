import base64
import io
import math
from typing import Literal

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from ..config import SAMPLE_RATE
from ..core.threader import threaded
from ..utils.signal_processing import bandpass_df

BandFilter = Literal["filtered", "delta", "theta", "alpha", "beta", "gamma", "raw"]


class VisualizationService:
    """Generate EEG visualizations"""

    BAND_FILTERS: dict[BandFilter, tuple[float, float] | tuple[None, None]] = {
        "filtered": (0.5, 40),
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 60),
        "raw": (None, None),
    }

    @threaded
    def visualize_df(self, df: pd.DataFrame, eeg_type: BandFilter) -> list[str]:
        """Visualize Dataframe as EEG plot and returns a single-element list with a base64 image.
        
        All electrodes are rendered in one combined figure.
        """

        low, high = self.BAND_FILTERS.get(eeg_type, (None, None))

        ch_names = df.select_dtypes(include=[np.number]).columns.tolist()
        ch_types = ["eeg"] * len(ch_names)

        # Apply bandpass filter
        df = bandpass_df(df, low=low, high=high)
        df_array = df.to_numpy().T

        # Create MNE info and raw object
        info = mne.create_info(ch_names=ch_names, sfreq=SAMPLE_RATE, ch_types=ch_types)
        _, sample_count = df_array.shape
        duration = sample_count / SAMPLE_RATE

        raw = mne.io.RawArray(df_array, info)
        raw_filt = raw.copy().filter(
            l_freq=low,
            h_freq=high,
            fir_design="firwin",
            phase="zero"
        )

        num_channels = len(ch_names)

        # Plot all channels in a single figure
        buf = io.BytesIO()
        fig = raw_filt.plot(
            scalings="auto",
            duration=30.0,
            n_channels=num_channels,
            show=False,
            show_scrollbars=False,
            clipping=None,
            block=False,
        )

        # Scale height proportionally to channel count
        fig.set_size_inches(math.ceil(duration / 6), max(num_channels * 0.6, 4))
        fig.savefig(buf, format="png", dpi=600)
        plt.close(fig)
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return [f"data:image/png;base64,{img_base64}"]
