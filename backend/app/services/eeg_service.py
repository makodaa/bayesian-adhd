from typing import Any, Generator, Literal

import mne
import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.integrate import simpson

from ..config import (
    BROADBAND,
    CLASSIFYING_FREQUENCY_BANDS,
    DISPLAY_FREQUENCY_BANDS,
    ELECTRODE_CHANNELS,
    MODEL_FREQUENCY_BANDS,
    SAMPLE_RATE,
    SAMPLES_PER_WINDOW,
    TARGET_FREQUENCY_BINS,
    WINDOW_STEP,
)
from ..core.logging_config import get_app_logger
from ..db.repositories.recordings import RecordingsRepository
from ..db.repositories.results import ResultsRepository
from ..ml.model_loader import ModelLoader

logger = get_app_logger(__name__)


class EEGService:
    """Handle EEG signal classification, ADHD prediction, and result saving"""

    def __init__(
        self,
        model_loader: ModelLoader,
        results_repo: ResultsRepository,
        recordings_repo: RecordingsRepository,
    ):
        self.model_loader = model_loader
        self.results_repo = results_repo
        self.recordings_repo = recordings_repo

    def process_window(self, window: pd.DataFrame, window_count: int):
        """Process a single window and compute frequency features."""
        logger.debug(f"Processing window {window_count} with {len(window)} samples")

        electrode_columns = (
            window[ELECTRODE_CHANNELS].select_dtypes(include=[np.number]).columns
        )

        n = len(window)
        assert len(electrode_columns) == 19, (
            f"There should only be 19 electrode columns, received {n}"
        )
        original_freqs = np.fft.rfftfreq(n, d=1 / SAMPLE_RATE)

        # compute power spectra for all electrodes
        electrode_powers = {}
        for electrode in electrode_columns:
            signal = window[electrode].to_numpy()
            fft_vals = np.fft.rfft(signal)
            power = np.abs(fft_vals) ** 2

            # interpolate to common freq bins
            electrode_powers[electrode] = np.interp(
                TARGET_FREQUENCY_BINS, original_freqs, power
            )

        # build rows: one per frequency bin
        for i, f in enumerate(TARGET_FREQUENCY_BINS):
            row = {
                "Window": window_count,
                "Frequency": f,
            }
            for electrode in electrode_columns:
                row[electrode] = electrode_powers[electrode][i]
            yield row

    def apply_ica_cleaning(self, df: pd.DataFrame, sfreq: float):
        """Apply ICA to remove muscle and eye artifacts from EEG dataframe."""

        ch_names = df.columns.tolist()
        ch_types = ["eeg"] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(df.to_numpy().T, info, verbose=False)

        # --- Filter for ICA (1–80 Hz typical) ---
        raw.filter(1.0, 60.0, fir_design="firwin", verbose=False)

        # --- Fit ICA (fixed component count: n_channels - 1) ---
        ica = mne.preprocessing.ICA(n_components=len(ch_names) - 1, method="fastica", verbose=False)
        ica.fit(raw, picks="eeg", decim=3)

        # --- Detect EOG-like (blink) components if present ---
        try:
            eog_inds, _ = ica.find_bads_eog(raw)
        except Exception:
            eog_inds = []

        # --- Detect muscle-like components (high-frequency ratio heuristic) ---
        sources = ica.get_sources(raw).get_data()
        n_ic = sources.shape[0]
        sfreq = raw.info["sfreq"]

        muscle_candidates = []
        for ic_idx in range(n_ic):
            f, Pxx = signal.welch(sources[ic_idx], sfreq, nperseg=512)
            hf_mask = (f >= 20)
            hf_ratio = Pxx[hf_mask].sum() / np.sum(Pxx)
            if hf_ratio > 0.25:  # threshold – tune as needed
                muscle_candidates.append(ic_idx)

        exclude = list(set(eog_inds + muscle_candidates))
        ica.exclude = exclude
        if len(exclude) > 0:
            print(f"ICA removed components: {exclude}")
            raw_clean = ica.apply(raw.copy())
        else:
            print("No ICA components removed.")
            raw_clean = raw

        # --- Convert back to pandas DataFrame ---
        cleaned = pd.DataFrame(raw_clean.get_data().T, columns=ch_names)
        return cleaned

    def apply_filter(self, df: pd.DataFrame, lower=0.5, upper=30):
        electrode_columns = df.select_dtypes(include=[np.number]).columns

        n = len(df)
        freqs = np.fft.rfftfreq(n, d=1 / SAMPLE_RATE)
        filtered_df = df.copy()

        for electrode in electrode_columns:
            signal = df[electrode].to_numpy()
            fft_vals = np.fft.rfft(signal)
            fft_filter = (freqs >= lower) & (freqs <= upper)
            fft_vals[~fft_filter] = 0
            filtered_signal = np.fft.irfft(fft_vals, n=n)
            filtered_df[electrode] = filtered_signal

        return filtered_df

    def apply_ica_cleaning_to_dataset(self, df: pd.DataFrame):
        result_df = df.copy()
        numerical_df = df.select_dtypes(include=[np.number])
        cleaned_df = self.apply_ica_cleaning(numerical_df, SAMPLE_RATE)
        for col in numerical_df.columns:
            result_df[col] = cleaned_df[col]

        return cleaned_df

    def compute_band_power(self, data, fs, band):
        """Compute absolute power in a frequency band using Welch's method"""
        low, high = band

        # Compute power spectral density using Welch's method
        freqs, psd = signal.welch(data, fs, nperseg=min(256, len(data)))

        # Find indices of frequencies in the band
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integrate power in the band using Simpson's rule
        band_power = simpson(psd[idx_band], freqs[idx_band])

        return band_power

    def compute_relative_band_powers(
        self, channel_data: np.ndarray, fs: float
    ) -> dict[str, float]:
        """Compute relative band power for all model bands for a single channel.

        Relative power = band_power / total_broadband_power
        where total_broadband_power is integrated over BROADBAND.
        A single Welch PSD is computed and reused for all bands.
        """
        freqs, psd = signal.welch(
            channel_data, fs, nperseg=min(256, len(channel_data))
        )

        # Total broadband power (denominator)
        bb_mask = (freqs >= BROADBAND[0]) & (freqs <= BROADBAND[1])
        total_power = simpson(psd[bb_mask], freqs[bb_mask])

        rel_powers: dict[str, float] = {}
        for band_name, (low, high) in MODEL_FREQUENCY_BANDS.items():
            idx = (freqs >= low) & (freqs <= high)
            abs_power = simpson(psd[idx], freqs[idx])
            rel_powers[band_name] = (
                abs_power / total_power if total_power > 0 else np.nan
            )

        return rel_powers

    def classify(self, df: pd.DataFrame) -> tuple[str, float, dict, list[dict]]:
        """Classify EEG data for ADHD probability.

        Returns:
            (classification, confidence, band_data, window_predictions)
            where window_predictions is a list of dicts with per-window results.
        """
        logger.info(f"Starting EEG classification on {len(df)} samples")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")

        # Check if columns are numeric (0-18 or 1-19) and rename them to electrode names
        if list(df.columns) == [str(i) for i in range(19)]:
            logger.info(
                "Detected numeric column names (0-18), renaming to electrode channels"
            )
            df.columns = ELECTRODE_CHANNELS
        elif list(df.columns) == list(range(19)):
            logger.info(
                "Detected numeric column names (0-18), renaming to electrode channels"
            )
            df.columns = ELECTRODE_CHANNELS
        elif list(df.columns) == [str(i) for i in range(1, 20)]:
            logger.info(
                "Detected numeric column names (1-19), renaming to electrode channels"
            )
            df.columns = ELECTRODE_CHANNELS
        elif list(df.columns) == list(range(1, 20)):
            logger.info(
                "Detected numeric column names (1-19), renaming to electrode channels"
            )
            df.columns = ELECTRODE_CHANNELS

        # Check if expected columns exist
        missing_channels = [ch for ch in ELECTRODE_CHANNELS if ch not in df.columns]
        if missing_channels:
            logger.error(f"Missing electrode channels: {missing_channels}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            raise ValueError(
                f"CSV file is missing required electrode channels: {missing_channels}. "
                f"Expected channels: {ELECTRODE_CHANNELS}. "
                f"Found columns: {df.columns.tolist()}"
            )
        window_count = 0
        n_samples = len(df)
        output = []

        # From now on, the subset_df is just the electrodes.
        df = df[ELECTRODE_CHANNELS]
        df = self.apply_filter(df, 0.5, 30)
        df = self.apply_ica_cleaning_to_dataset(df)

        n_samples = len(df)

        # Compute relative band powers for each channel and average
        rel_powers = {band: 0.0 for band in MODEL_FREQUENCY_BANDS}

        for channel in ELECTRODE_CHANNELS:
            if channel in df.columns:
                ch_rel = self.compute_relative_band_powers(
                    df[channel].values, SAMPLE_RATE
                )
                for band_name in MODEL_FREQUENCY_BANDS:
                    rel_powers[band_name] += ch_rel[band_name]

        # Average across channels
        n_channels = len(ELECTRODE_CHANNELS)
        for band_name in rel_powers:
            rel_powers[band_name] /= n_channels

        # Compute ratios from relative powers
        tbr = rel_powers["theta"] / rel_powers["beta"] if rel_powers["beta"] else np.nan
        tar = rel_powers["theta"] / rel_powers["alpha"] if rel_powers["alpha"] else np.nan

        # Prepare band data for return
        band_data = {
            "average_relative_power": {
                band: float(power) for band, power in rel_powers.items()
            },
            "band_ratios": {
                "theta_beta_ratio": float(tbr) if not np.isnan(tbr) else 0.0,
                "theta_alpha_ratio": float(tar) if not np.isnan(tar) else 0.0,
            },
        }

        # Store relative power results for model features (matching training notebook)
        power_results = {
            "theta": rel_powers["theta"],
            "beta": rel_powers["beta"],
            "alpha": rel_powers["alpha"],
            "fast_alpha": rel_powers["fast_alpha"],
            "high_beta": rel_powers["high_beta"],
            "tbr": tbr,
            "tar": tar,
        }

        # Normalize the data subject-wise.
        std_dev = df.std()
        mean = df.mean()
        df = (df - mean) / std_dev

        # Sliding window with overlap
        # Takes WINDOW_STEP steps every iteration
        window_sample_ranges: list[tuple[int, int]] = []
        for start in range(0, n_samples - SAMPLES_PER_WINDOW + 1, WINDOW_STEP):
            window = df.iloc[start : start + SAMPLES_PER_WINDOW]
            window_sample_ranges.append((start, start + SAMPLES_PER_WINDOW))
            for frequency in self.process_window(window, window_count):
                output.append(frequency)
            window_count += 1

        logger.info(f"Processed {window_count} windows from EEG data")

        # Create frequency dataframe and add power columns
        frequency_df = pd.DataFrame(output)
        for power_name, power_value in power_results.items():
            frequency_df[power_name] = power_value

        frequency_count = len(frequency_df["Frequency"].unique())
        window_count = len(frequency_df["Window"].unique())
        numeric_df = frequency_df.drop(["Window", "Frequency"], axis=1)

        # shape: (windows, freq, features)
        # Reshaping dataframe into tensor for classification
        logger.debug(
            f"Reshaping data: {window_count} windows, {frequency_count} frequencies, {numeric_df.shape[1]} features"
        )
        full_ndarray = numeric_df.values.reshape(
            (window_count, frequency_count, numeric_df.shape[1])
        )
        full_ndarray = full_ndarray[..., np.newaxis]

        logger.info("Running model inference")
        with torch.no_grad():
            tensor = torch.tensor(full_ndarray, dtype=torch.float32).permute(0, 3, 1, 2)

            print(f"The tensor shape: {tensor.shape}")
            x_eeg = tensor[:, :, :, :19]  # (1, 77, 19)
            x_band = tensor[:, 0, 0, 19:]  # (7,)

            print((x_eeg.shape, x_band.shape))
            predictions = (
                self.model_loader.model(x_eeg, x_band).softmax(1).detach().numpy()
            )
            print("=" * 100)
            print(f"Predictions: {predictions}")
            print("=" * 100)

            class_names = [
                "Combined / C (ADHD-C)",
                "Hyperactive-Impulsive (ADHD-H)",
                "Inattentive (ADHD-I)",
                "Non-ADHD",
            ]

            # Build per-window prediction list
            window_predictions: list[dict] = []
            for w_idx in range(predictions.shape[0]):
                w_probs = predictions[w_idx]  # softmax probabilities
                w_argmax = int(np.argmax(w_probs))
                start_s, end_s = window_sample_ranges[w_idx]
                window_predictions.append({
                    "window": w_idx,
                    "start_sample": start_s,
                    "end_sample": end_s,
                    "start_time": round(start_s / SAMPLE_RATE, 4),
                    "end_time": round(end_s / SAMPLE_RATE, 4),
                    "predicted_class": class_names[w_argmax],
                    "confidence": float(w_probs[w_argmax]),
                    "probabilities": {
                        class_names[i]: float(w_probs[i]) for i in range(4)
                    },
                })

            logger.info(f"Per-window predictions computed for {len(window_predictions)} windows")

            adhd_1, adhd_2, adhd_3, control = np.sum(predictions, axis=0) / np.sum(
                predictions
            )
            maximum = max(adhd_1, adhd_2, adhd_3, control)
            # Store confidence as decimal (0-1) - frontend/PDF will multiply by 100 for display
            conf = float(maximum)

            class_probs = [adhd_1, adhd_2, adhd_3, control]
            overall_idx = int(np.argmax(class_probs))
            overall_name = class_names[overall_idx]

            logger.info(
                f"Classification result: {overall_name} with {conf * 100:.2f}% confidence"
            )
            return overall_name, conf, band_data, window_predictions

    def classify_and_save(
        self,
        recording_id: int,
        df: pd.DataFrame,
        clinician_id: int | None = None,
        vanderbilt_result: dict[str, Any] | None = None,
        vanderbilt_symptom_scores: dict[str, int] | None = None,
        vanderbilt_performance_scores: dict[str, int] | None = None,
    ) -> dict:
        """Classify EEG data and save results to database."""
        logger.info(f"Starting classify and save for recording {recording_id}")
        classification, confidence, band_data, window_predictions = self.classify(df)

        logger.info(
            f"Saving classification result to database: {classification} ({confidence * 100:.2f}%)"
        )
        result_id = self.results_repo.create_result(
            recording_id=recording_id,
            classification=classification,
            confidence_score=confidence,
            clinician_id=clinician_id,
            vanderbilt_scale_type=(
                str(vanderbilt_result["scale_type"])
                if vanderbilt_result and "scale_type" in vanderbilt_result
                else None
            ),
            vanderbilt_inattentive_count=(
                int(vanderbilt_result["inattentive_count"])
                if vanderbilt_result and "inattentive_count" in vanderbilt_result
                else None
            ),
            vanderbilt_hyperactive_impulsive_count=(
                int(vanderbilt_result["hyperactive_impulsive_count"])
                if vanderbilt_result and "hyperactive_impulsive_count" in vanderbilt_result
                else None
            ),
            vanderbilt_performance_impairment_count=(
                int(vanderbilt_result["performance_impairment_count"])
                if vanderbilt_result and "performance_impairment_count" in vanderbilt_result
                else None
            ),
            vanderbilt_adhd_inattentive_met=(
                bool(vanderbilt_result["adhd_inattentive_criteria_met"])
                if vanderbilt_result and "adhd_inattentive_criteria_met" in vanderbilt_result
                else None
            ),
            vanderbilt_adhd_hyperactive_impulsive_met=(
                bool(vanderbilt_result["adhd_hyperactive_impulsive_criteria_met"])
                if vanderbilt_result
                and "adhd_hyperactive_impulsive_criteria_met" in vanderbilt_result
                else None
            ),
            vanderbilt_adhd_combined_met=(
                bool(vanderbilt_result["adhd_combined_criteria_met"])
                if vanderbilt_result and "adhd_combined_criteria_met" in vanderbilt_result
                else None
            ),
            vanderbilt_interpretation=(
                str(vanderbilt_result["interpretation"])
                if vanderbilt_result and "interpretation" in vanderbilt_result
                else None
            ),
            vanderbilt_domain_scores=(
                {
                    "domains": vanderbilt_result.get("domains", {}),
                    "criteria_outcome": vanderbilt_result.get("criteria_outcome"),
                }
                if vanderbilt_result
                else None
            ),
            vanderbilt_symptom_scores=vanderbilt_symptom_scores,
            vanderbilt_performance_scores=vanderbilt_performance_scores,
        )

        logger.info(
            f"Classification complete for recording {recording_id}, result ID: {result_id}"
        )
        return {
            "recording_id": recording_id,
            "result_id": result_id,
            "classification": classification,
            "confidence_score": confidence,
            "clinician_id": clinician_id,
            "window_predictions": window_predictions,
            "vanderbilt": vanderbilt_result,
        }
