from __future__ import annotations

from typing import Iterable, TypedDict

import numpy as np
import pandas as pd
from scipy.signal import welch


class BandPowerResults(TypedDict):
    absolute_power: dict[str, dict[str, float]]
    relative_power: dict[str, dict[str, float]]
    total_power: dict[str, float]


def compute_electrode_band_powers(
    df: pd.DataFrame,
    bands: dict[str, tuple[float, float]],
    sample_rate: float,
    electrodes: Iterable[str] | None = None,
    nperseg: int | None = None,
) -> BandPowerResults:
    """Compute per-electrode absolute/relative band powers.

    Returns a dict with keys: absolute_power, relative_power, total_power.
    """
    if electrodes is None:
        electrode_list: list[str] = (
            df.select_dtypes(include=[np.number]).columns.tolist()
        )
    else:
        electrode_list = list(electrodes)

    absolute_by_electrode: dict[str, dict[str, float]] = {}
    relative_by_electrode: dict[str, dict[str, float]] = {}
    total_by_electrode: dict[str, float] = {}

    for electrode in electrode_list:
        signal = df[electrode].to_numpy()

        if nperseg is None:
            nperseg_local = min(256, len(signal) // 4)
        else:
            nperseg_local = nperseg

        freqs, psd = welch(signal, fs=sample_rate, nperseg=nperseg_local)

        absolute_powers: dict[str, float] = {}
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs < high)
            band_power = np.trapezoid(psd[band_mask], freqs[band_mask])
            absolute_powers[band_name] = float(band_power)

        total_power = sum(absolute_powers.values())
        relative_powers = {
            band_name: (power / total_power) if total_power > 0 else 0.0
            for band_name, power in absolute_powers.items()
        }

        absolute_by_electrode[electrode] = absolute_powers
        relative_by_electrode[electrode] = relative_powers
        total_by_electrode[electrode] = float(total_power)

    return {
        "absolute_power": absolute_by_electrode,
        "relative_power": relative_by_electrode,
        "total_power": total_by_electrode,
    }
