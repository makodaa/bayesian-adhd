"""
Topographic scalp heatmap generation service.

Generates MNE topomap visualizations for:
- Absolute power per electrode per band
- Relative power per electrode per band
- Theta/Beta ratio (TBR) per electrode
"""

import base64
import io
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from ..config import CLASSIFYING_FREQUENCY_BANDS, DISPLAY_FREQUENCY_BANDS, ELECTRODE_CHANNELS, SAMPLE_RATE
from ..core.logging_config import get_app_logger

matplotlib.use("Agg")

logger = get_app_logger(__name__)

# Standard 10-20 montage electrode names (must match MNE naming)
# Our ELECTRODE_CHANNELS use T7/T8/P7/P8 which are the modern 10-20 names
# MNE's standard_1020 montage supports these directly

TopoMapType = Literal["absolute", "relative", "tbr"]

# Band display colors for consistency
BAND_CMAPS = {
    "delta": "Purples",
    "theta": "Blues",
    "alpha": "Greens",
    "beta": "Oranges",
    "gamma": "Reds",
}


class TopographicService:
    """Generate topographic scalp heatmaps from EEG band power data."""

    def __init__(self):
        self._montage = mne.channels.make_standard_montage("standard_1020")

    def _create_info(self, ch_names: list[str]) -> mne.Info:
        """Create MNE Info object with electrode positions from standard 10-20."""
        info = mne.create_info(ch_names=ch_names, sfreq=SAMPLE_RATE, ch_types="eeg")
        info.set_montage(self._montage, on_missing="warn")
        return info

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert a matplotlib figure to a base64-encoded PNG data URI."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def compute_electrode_band_powers(self, df: pd.DataFrame) -> dict:
        """
        Compute per-electrode absolute & relative band powers and TBR.

        Returns
        -------
        dict with keys:
            - absolute_power: {electrode: {band: float}}
            - relative_power: {electrode: {band: float}}
            - tbr: {electrode: float}  (theta/beta ratio per electrode)
        """
        electrodes = [c for c in df.select_dtypes(include=[np.number]).columns
                      if c in ELECTRODE_CHANNELS]

        result = {
            "absolute_power": {},
            "relative_power": {},
            "tbr": {},
        }

        for electrode in electrodes:
            signal = df[electrode].to_numpy()
            freqs, psd = welch(signal, fs=SAMPLE_RATE, nperseg=min(256, len(signal) // 4))

            absolute_powers = {}
            for band_name, (low, high) in DISPLAY_FREQUENCY_BANDS.items():
                band_mask = (freqs >= low) & (freqs < high)
                band_power = float(np.trapezoid(psd[band_mask], freqs[band_mask]))
                absolute_powers[band_name] = band_power

            total_power = sum(absolute_powers.values())
            relative_powers = {
                band: (power / total_power) if total_power > 0 else 0.0
                for band, power in absolute_powers.items()
            }

            result["absolute_power"][electrode] = absolute_powers
            result["relative_power"][electrode] = relative_powers

            # Theta/Beta ratio per electrode
            theta_power = absolute_powers.get("theta", 0.0)
            beta_power = absolute_powers.get("beta", 0.0)
            result["tbr"][electrode] = float(theta_power / beta_power) if beta_power > 0 else 0.0

        return result

    def generate_band_topomap(
        self,
        electrode_values: dict[str, float],
        title: str,
        cmap: str = "RdYlBu_r",
        unit: str = "",
    ) -> str:
        """
        Generate a single topographic map image.

        Parameters
        ----------
        electrode_values : dict mapping electrode name -> scalar value
        title : str for the plot title
        cmap : matplotlib colormap name
        unit : label for the colorbar

        Returns
        -------
        str : base64-encoded PNG data URI
        """
        ch_names = [ch for ch in ELECTRODE_CHANNELS if ch in electrode_values]
        if not ch_names:
            logger.warning("No valid electrodes found for topomap generation")
            return ""

        values = np.array([electrode_values[ch] for ch in ch_names])
        info = self._create_info(ch_names)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        im, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=ax,
            cmap=cmap,
            show=False,
            contours=6,
            sensors=True,
            names=ch_names,
        )
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if unit:
            cbar.set_label(unit, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

        return self._fig_to_base64(fig)

    def generate_all_topomaps(self, df: pd.DataFrame) -> dict:
        """
        Generate all topographic heatmaps from raw EEG data.

        Returns
        -------
        dict with keys:
            - absolute_power_maps: {band: base64_image}
            - relative_power_maps: {band: base64_image}
            - tbr_map: base64_image
            - electrode_data: raw numeric data used for maps
        """
        logger.info("Generating topographic heatmaps")
        powers = self.compute_electrode_band_powers(df)

        absolute_maps = {}
        relative_maps = {}

        for band in DISPLAY_FREQUENCY_BANDS:
            cmap = BAND_CMAPS.get(band, "RdYlBu_r")

            # Absolute power topomap
            abs_values = {
                elec: powers["absolute_power"][elec][band]
                for elec in powers["absolute_power"]
            }
            absolute_maps[band] = self.generate_band_topomap(
                abs_values,
                title=f"{band.capitalize()} - Absolute Power",
                cmap=cmap,
                unit="µV²/Hz",
            )

            # Relative power topomap
            rel_values = {
                elec: powers["relative_power"][elec][band]
                for elec in powers["relative_power"]
            }
            relative_maps[band] = self.generate_band_topomap(
                rel_values,
                title=f"{band.capitalize()} - Relative Power",
                cmap=cmap,
                unit="%",
            )

        # TBR topomap
        tbr_map = self.generate_band_topomap(
            powers["tbr"],
            title="Theta/Beta Ratio (TBR)",
            cmap="RdYlBu_r",
            unit="TBR",
        )

        logger.info(
            f"Generated {len(absolute_maps)} absolute + {len(relative_maps)} relative + 1 TBR topomaps"
        )

        return {
            "absolute_power_maps": absolute_maps,
            "relative_power_maps": relative_maps,
            "tbr_map": tbr_map,
            "electrode_data": {
                "absolute_power": powers["absolute_power"],
                "relative_power": powers["relative_power"],
                "tbr": powers["tbr"],
            },
        }

    def generate_topomaps_from_db(self, band_powers: list[dict]) -> dict:
        """
        Generate topographic heatmaps from stored band power records.

        Parameters
        ----------
        band_powers : list of dicts from BandPowersRepository.get_by_result()
            Each dict has: electrode, frequency_band, absolute_power, relative_power

        Returns
        -------
        dict : same structure as generate_all_topomaps
        """
        logger.info(f"Generating topomaps from {len(band_powers)} stored band power records")

        # Reorganize DB records into electrode->band->power structure
        abs_by_electrode: dict[str, dict[str, float]] = {}
        rel_by_electrode: dict[str, dict[str, float]] = {}

        for record in band_powers:
            elec = record["electrode"]
            band = record["frequency_band"]

            if band not in DISPLAY_FREQUENCY_BANDS:
                continue

            if elec not in abs_by_electrode:
                abs_by_electrode[elec] = {}
                rel_by_electrode[elec] = {}

            abs_by_electrode[elec][band] = float(record["absolute_power"])
            rel_by_electrode[elec][band] = float(record["relative_power"])

        # Compute per-electrode TBR from the stored absolute powers
        tbr_values = {}
        for elec in abs_by_electrode:
            theta = abs_by_electrode[elec].get("theta", 0.0)
            beta = abs_by_electrode[elec].get("beta", 0.0)
            tbr_values[elec] = float(theta / beta) if beta > 0 else 0.0

        absolute_maps = {}
        relative_maps = {}

        for band in DISPLAY_FREQUENCY_BANDS:
            cmap = BAND_CMAPS.get(band, "RdYlBu_r")

            abs_values = {elec: abs_by_electrode[elec].get(band, 0.0) for elec in abs_by_electrode}
            absolute_maps[band] = self.generate_band_topomap(
                abs_values,
                title=f"{band.capitalize()} - Absolute Power",
                cmap=cmap,
                unit="µV²/Hz",
            )

            rel_values = {elec: rel_by_electrode[elec].get(band, 0.0) for elec in rel_by_electrode}
            relative_maps[band] = self.generate_band_topomap(
                rel_values,
                title=f"{band.capitalize()} - Relative Power",
                cmap=cmap,
                unit="%",
            )

        tbr_map = self.generate_band_topomap(
            tbr_values,
            title="Theta/Beta Ratio (TBR)",
            cmap="RdYlBu_r",
            unit="TBR",
        )

        return {
            "absolute_power_maps": absolute_maps,
            "relative_power_maps": relative_maps,
            "tbr_map": tbr_map,
            "electrode_data": {
                "absolute_power": abs_by_electrode,
                "relative_power": rel_by_electrode,
                "tbr": tbr_values,
            },
        }
