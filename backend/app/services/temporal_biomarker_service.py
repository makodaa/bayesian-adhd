"""
Temporal biomarker evolution service.

Computes EEG biomarkers across sliding time windows to show how they
evolve throughout a recording.  The resulting time-series are plotted
and returned as base64-encoded images grouped by category.

Biomarker categories
--------------------
Band Ratios      : TBR, T/A, A/T, A/B, D/T, LowB/HighB, combined
Relative Powers  : relative delta, theta, alpha, beta, gamma
"""

from __future__ import annotations

import base64
import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from ..config import (
    CLASSIFYING_FREQUENCY_BANDS,
    ELECTRODE_CHANNELS,
    OVERLAP,
    SAMPLE_RATE,
    WINDOW_SECONDS,
)
from ..core.logging_config import get_app_logger

matplotlib.use("Agg")
logger = get_app_logger(__name__)

# ── Biomarker definitions (label, key) ──────────────────────────────────

BIOMARKER_DEFS: list[tuple[str, str]] = [
    # Band ratios
    ("Theta/Beta", "theta_beta_ratio"),
    ("Theta/Alpha", "theta_alpha_ratio"),
    ("Alpha/Theta", "alpha_theta_ratio"),
    ("Alpha/Beta", "alpha_beta_ratio"),
    ("Delta/Theta", "delta_theta_ratio"),
    ("Low Beta/High Beta", "low_beta_high_beta_ratio"),
    ("Combined (T+B)/(A+G)", "combined_ratio"),
    # Relative band powers
    ("Relative Delta", "relative_delta"),
    ("Relative Theta", "relative_theta"),
    ("Relative Alpha", "relative_alpha"),
    ("Relative Beta", "relative_beta"),
    ("Relative Gamma", "relative_gamma"),
]

# Logical plot groups → (group title, list of biomarker keys)
BIOMARKER_GROUPS: list[tuple[str, list[str]]] = [
    (
        "Band Power Ratios",
        [
            "theta_beta_ratio",
            "theta_alpha_ratio",
            "alpha_theta_ratio",
            "alpha_beta_ratio",
            "delta_theta_ratio",
            "low_beta_high_beta_ratio",
            "combined_ratio",
        ],
    ),
    (
        "Relative Band Powers",
        [
            "relative_delta",
            "relative_theta",
            "relative_alpha",
            "relative_beta",
            "relative_gamma",
        ],
    ),
]

# Quick label lookup
_KEY_TO_LABEL: dict[str, str] = {k: lbl for lbl, k in BIOMARKER_DEFS}

# Plot colour palette (enough for 7 lines per subplot)
_COLORS = [
    "#2196F3",
    "#F44336",
    "#4CAF50",
    "#FF9800",
    "#9C27B0",
    "#00BCD4",
    "#795548",
]


# ── Helper: band-power from a short signal segment ─────────────────────

def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


# ── Core per-window computation ────────────────────────────────────────


def _compute_window_biomarkers(
    window_data: pd.DataFrame,
    electrodes: list[str],
) -> dict[str, float]:
    """Compute ratios and relative band powers for a single time window."""

    n_samples = len(window_data)
    nperseg = min(128, max(16, n_samples // 2))

    # ── Per-electrode PSD ───────────────────────────────────────────
    band_powers: dict[str, dict[str, float]] = {}

    for elec in electrodes:
        sig = window_data[elec].to_numpy().astype(float)
        freqs, psd = welch(sig, fs=SAMPLE_RATE, nperseg=nperseg)

        powers: dict[str, float] = {}
        for band, (lo, hi) in CLASSIFYING_FREQUENCY_BANDS.items():
            powers[band] = _band_power(freqs, psd, lo, hi)
        band_powers[elec] = powers

    # ── Global average band powers ──────────────────────────────────
    avg_bp: dict[str, float] = {}
    for band in CLASSIFYING_FREQUENCY_BANDS:
        avg_bp[band] = float(np.mean([band_powers[e][band] for e in electrodes]))

    # ── 1. Band ratios ──────────────────────────────────────────────
    theta = avg_bp["theta"]
    beta = avg_bp["beta"]
    alpha = avg_bp["alpha"]
    delta = avg_bp["delta"]
    gamma = avg_bp["gamma"]
    high_beta = avg_bp.get("high_beta", 0.0)
    low_beta = beta - high_beta  # beta(13-30) minus high_beta(15-30) ≈ low_beta(13-15)
    if low_beta < 0:
        low_beta = 0.0

    biomarkers: dict[str, float] = {}
    biomarkers["theta_beta_ratio"] = _safe_ratio(theta, beta)
    biomarkers["theta_alpha_ratio"] = _safe_ratio(theta, alpha)
    biomarkers["alpha_theta_ratio"] = _safe_ratio(alpha, theta)
    biomarkers["alpha_beta_ratio"] = _safe_ratio(alpha, beta)
    biomarkers["delta_theta_ratio"] = _safe_ratio(delta, theta)
    biomarkers["low_beta_high_beta_ratio"] = _safe_ratio(low_beta, high_beta)
    biomarkers["combined_ratio"] = _safe_ratio(
        theta + beta, alpha + gamma
    ) if (alpha + gamma) > 0 else 0.0

    # ── 2. Relative band powers ─────────────────────────────────────
    total_power = delta + theta + alpha + beta + gamma
    if total_power > 0:
        biomarkers["relative_delta"] = float(delta / total_power)
        biomarkers["relative_theta"] = float(theta / total_power)
        biomarkers["relative_alpha"] = float(alpha / total_power)
        biomarkers["relative_beta"] = float(beta / total_power)
        biomarkers["relative_gamma"] = float(gamma / total_power)
    else:
        biomarkers["relative_delta"] = 0.0
        biomarkers["relative_theta"] = 0.0
        biomarkers["relative_alpha"] = 0.0
        biomarkers["relative_beta"] = 0.0
        biomarkers["relative_gamma"] = 0.0

    return biomarkers


# ── Service class ──────────────────────────────────────────────────────


class TemporalBiomarkerService:
    """Compute biomarker time-series and render evolution plots."""

    def compute_temporal_biomarkers(
        self,
        df: pd.DataFrame,
        window_sec: float = WINDOW_SECONDS,
        overlap: float = OVERLAP,
    ) -> dict:
        """
        Slide a window across the recording and compute biomarkers in each.

        Parameters
        ----------
        df : pd.DataFrame
            Raw EEG data (rows = samples, columns = electrode channels).
        window_sec : float
            Window size in seconds.
        overlap : float
            Fraction of overlap between consecutive windows.

        Returns
        -------
        dict
            - timestamps : list[float]   (centre time of each window in s)
            - biomarkers : dict[str, list[float]]  (key → time-series)
        """
        electrodes = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c in ELECTRODE_CHANNELS
        ]
        if not electrodes:
            raise ValueError("No recognised EEG electrode columns in data")

        n_samples = len(df)
        win_samples = int(window_sec * SAMPLE_RATE)
        step_samples = int(win_samples * (1 - overlap))

        if n_samples < win_samples:
            raise ValueError(
                f"Recording too short ({n_samples} samples) for a "
                f"{window_sec}s window ({win_samples} samples)"
            )

        timestamps: list[float] = []
        all_biomarkers: dict[str, list[float]] = {k: [] for _, k in BIOMARKER_DEFS}

        start = 0
        while start + win_samples <= n_samples:
            end = start + win_samples
            window_df = df.iloc[start:end][electrodes]

            centre_time = (start + win_samples / 2) / SAMPLE_RATE
            timestamps.append(round(centre_time, 3))

            bm = _compute_window_biomarkers(window_df, electrodes)
            for _, key in BIOMARKER_DEFS:
                all_biomarkers[key].append(bm.get(key, 0.0))

            start += step_samples

        logger.info(
            f"Computed {len(timestamps)} windows × {len(BIOMARKER_DEFS)} biomarkers"
        )

        return {
            "timestamps": timestamps,
            "biomarkers": all_biomarkers,
        }

    # ── Plotting ────────────────────────────────────────────────────

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

    def plot_biomarker_groups(
        self,
        timestamps: list[float],
        biomarkers: dict[str, list[float]],
    ) -> list[dict]:
        """
        Create one figure per biomarker group, each containing subplots.

        Returns
        -------
        list of  {group: str, image: str (base64)}
        """
        results: list[dict] = []

        for group_title, keys in BIOMARKER_GROUPS:
            valid_keys = [k for k in keys if k in biomarkers and biomarkers[k]]
            if not valid_keys:
                continue

            n_plots = len(valid_keys)
            fig, axes = plt.subplots(
                n_plots, 1,
                figsize=(10, 2.2 * n_plots),
                sharex=True,
                squeeze=False,
            )

            # Build human-readable time tick labels
            import matplotlib.ticker as mticker

            duration = timestamps[-1] if timestamps else 0
            def _fmt_time(x, _pos=None):
                mins, secs = divmod(int(x), 60)
                if duration >= 120:
                    return f"{mins}:{secs:02d}"
                return f"{x:.0f}s"

            for i, key in enumerate(valid_keys):
                ax = axes[i, 0]
                color = _COLORS[i % len(_COLORS)]
                ax.plot(timestamps, biomarkers[key], color=color, linewidth=1.2)
                ax.fill_between(timestamps, biomarkers[key], alpha=0.12, color=color)
                ax.set_ylabel(_KEY_TO_LABEL.get(key, key), fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                ax.margins(x=0)
                # Show time ticks on every subplot for clarity
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_time))
                ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

            axes[-1, 0].set_xlabel(
                "Time (m:ss)" if duration >= 120 else "Time (seconds)",
                fontsize=9,
            )
            fig.suptitle(group_title, fontsize=12, fontweight="bold", y=1.01)
            fig.tight_layout()

            results.append({
                "group": group_title,
                "image": self._fig_to_base64(fig),
            })

        return results

    # ── Public high-level API ───────────────────────────────────────

    def generate_temporal_plots(self, df: pd.DataFrame) -> dict:
        """
        End-to-end: compute biomarkers → plot → return images + raw data.

        Returns
        -------
        dict
            - plots : list[{group, image}]
            - timestamps : list[float]
            - biomarkers : dict[str, list[float]]
            - summary : dict[str, {mean, std, min, max}]
        """
        data = self.compute_temporal_biomarkers(df)
        plots = self.plot_biomarker_groups(data["timestamps"], data["biomarkers"])

        # Compute summary statistics for each biomarker
        summary: dict[str, dict[str, float]] = {}
        for _, key in BIOMARKER_DEFS:
            vals = data["biomarkers"].get(key, [])
            if vals:
                arr = np.array(vals)
                summary[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }

        return {
            "plots": plots,
            "timestamps": data["timestamps"],
            "biomarkers": data["biomarkers"],
            "summary": summary,
        }
