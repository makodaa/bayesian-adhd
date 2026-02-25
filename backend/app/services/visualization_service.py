import io
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from ..config import SAMPLE_RATE
from ..core.logging_config import get_app_logger

BandFilter = Literal["filtered", "delta", "theta", "alpha", "beta", "gamma", "raw"]

logger = get_app_logger(__name__)


class VisualizationService:
    """Generate lightweight EEG preview visualizations."""

    BAND_FILTERS: dict[BandFilter, tuple[float, float] | tuple[None, None]] = {
        "filtered": (0.5, 40),
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 60),
        "raw": (None, None),
    }

    PREVIEW_DPI = 130
    PREVIEW_MAX_POINTS = 2500

    @staticmethod
    def _downsample(data: np.ndarray, max_points: int) -> tuple[np.ndarray, int]:
        sample_count = data.shape[0]
        if sample_count <= max_points:
            return data, 1

        step = int(np.ceil(sample_count / max_points))
        return data[::step], step

    @staticmethod
    def _normalize_signals(data: np.ndarray) -> np.ndarray:
        centered = data - np.mean(data, axis=0, keepdims=True)
        scales = np.std(centered, axis=0, keepdims=True)
        scales[scales < 1e-8] = 1.0
        return centered / scales

    @staticmethod
    def _apply_filter(data: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
        if low is None and high is None:
            return data

        nyquist = SAMPLE_RATE / 2
        low_cut = (low / nyquist) if low is not None else None
        high_cut = (high / nyquist) if high is not None else None

        if low_cut is not None and low_cut <= 0:
            low_cut = None
        if high_cut is not None and high_cut >= 1.0:
            high_cut = 0.99

        if low_cut is not None and high_cut is not None:
            filter_type = "bandpass"
            cutoff = [low_cut, high_cut]
        elif low_cut is not None:
            filter_type = "highpass"
            cutoff = low_cut
        elif high_cut is not None:
            filter_type = "lowpass"
            cutoff = high_cut
        else:
            return data

        try:
            sos = butter(4, cutoff, btype=filter_type, output="sos")
            return sosfiltfilt(sos, data, axis=0).astype(np.float32, copy=False)
        except ValueError:
            logger.warning("Signal too short for preview filtering; returning unfiltered data")
            return data

    def render_preview_png(self, df: pd.DataFrame, eeg_type: BandFilter) -> bytes:
        """Render a low-cost stacked EEG preview and return PNG bytes."""
        if eeg_type not in self.BAND_FILTERS:
            raise ValueError("Invalid EEG type")

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric EEG channels found")

        channel_names = numeric_df.columns.tolist()
        signal_data = numeric_df.to_numpy(dtype=np.float32, copy=True)

        low, high = self.BAND_FILTERS[eeg_type]
        filtered = self._apply_filter(signal_data, low, high)
        reduced, step = self._downsample(filtered, self.PREVIEW_MAX_POINTS)
        normalized = self._normalize_signals(reduced)

        sample_count, channel_count = normalized.shape
        time_axis = (
            np.arange(sample_count, dtype=np.float32) * (step / float(SAMPLE_RATE))
        )

        offsets = np.arange(channel_count, dtype=np.float32) * 3.0
        height = min(max(4.0, 1.2 + channel_count * 0.32), 14.0)
        width = min(max(10.0, time_axis[-1] / 5.0 if len(time_axis) > 1 else 10.0), 18.0)

        fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=self.PREVIEW_DPI)
        for idx in range(channel_count):
            ax.plot(
                time_axis,
                normalized[:, idx] + offsets[idx],
                linewidth=0.6,
                color="#1f77b4",
                alpha=0.9,
            )

        ax.set_yticks(offsets)
        ax.set_yticklabels(channel_names, fontsize=7)
        ax.set_xlabel("Time (seconds)", fontsize=8)
        ax.set_title(f"{eeg_type.capitalize()} EEG Preview", fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        ax.margins(x=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if len(time_axis) > 1:
            ax.set_xlim(float(time_axis[0]), float(time_axis[-1]))

        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=self.PREVIEW_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        buffer.seek(0)
        return buffer.read()
