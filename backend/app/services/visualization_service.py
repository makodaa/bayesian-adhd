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
        """Visualize Dataframe as EEG plot and returns a list of base64 images.
        
        Splits the electrodes into 6 groups with even numbers to avoid overlap.
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

        # Split electrodes into 6 groups with even numbers (2-4 electrodes each)
        # 19 electrodes: [4, 4, 4, 3, 2, 2] = 19
        num_channels = len(ch_names)
        num_groups = 6
        base_size = num_channels // num_groups  # 19 // 6 = 3
        remainder = num_channels % num_groups   # 19 % 6 = 1
        
        # Create group sizes: distribute remainder to first groups
        group_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_groups)]
        # Adjust to ensure even numbers where possible
        group_sizes = [4, 4, 4, 3, 2, 2]  # For 19 electrodes
        
        images = []
        start_idx = 0
        
        for group_size in group_sizes:
            if start_idx >= num_channels:
                break
                
            end_idx = min(start_idx + group_size, num_channels)
            group_channels = ch_names[start_idx:end_idx]
            
            # Pick only the channels for this group
            raw_group = raw_filt.copy().pick_channels(group_channels)
            
            # Plot this group
            buf = io.BytesIO()
            fig = raw_group.plot(
                scalings="auto",  # 100 microvolts - allows signal variation to be visible
                duration=30.0,
                n_channels=len(group_channels),
                show=False,
                show_scrollbars=False,
                clipping=None,
                block=False,
            )

            fig.set_size_inches(math.ceil(duration / 6), 3)
            fig.savefig(buf, format="png", dpi=600)
            plt.close(fig)
            buf.seek(0)

            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            images.append(f"data:image/png;base64,{img_base64}")
            
            start_idx = end_idx
        
        return images
