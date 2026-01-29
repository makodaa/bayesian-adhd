from typing import Any, Generator, Literal
import mne
import torch
import numpy as np
import pandas as pd
from ..ml.model_loader import ModelLoader
from ..config import ELECTRODE_CHANNELS, SAMPLE_RATE, SAMPLES_PER_WINDOW, WINDOW_STEP, TARGET_FREQUENCY_BINS, FREQUENCY_BANDS
from ..db.repositories.results import ResultsRepository
from ..db.repositories.recordings import RecordingsRepository
from ..core.logging_config import get_app_logger
from scipy import signal
from scipy.integrate import simpson

logger = get_app_logger(__name__)


class EEGService:
    """Handle EEG signal classification, ADHD prediction, and result saving"""
    def __init__(self,
                model_loader:ModelLoader,
                results_repo:ResultsRepository,
                recordings_repo:RecordingsRepository):
        self.model_loader = model_loader
        self.results_repo = results_repo
        self.recordings_repo = recordings_repo

    def process_window(self, window:pd.DataFrame, window_count:int) :
        """Process a single window and compute frequency features."""
        logger.debug(f"Processing window {window_count} with {len(window)} samples")

        electrode_columns = window[ELECTRODE_CHANNELS].select_dtypes(include=[np.number]).columns

        n = len(window)
        assert len(electrode_columns) == 19, f"There should only be 19 electrode columns, received {n}"
        original_freqs = np.fft.rfftfreq(n, d=1 / SAMPLE_RATE)

        # compute power spectra for all electrodes
        electrode_powers = {}
        for electrode in electrode_columns:
            signal = window[electrode].to_numpy()
            fft_vals = np.fft.rfft(signal)
            power = np.abs(fft_vals) ** 2

            # interpolate to common freq bins
            electrode_powers[electrode] = np.interp(TARGET_FREQUENCY_BINS, original_freqs, power)

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

        # --- Fit ICA ---
        ica = mne.preprocessing.ICA(n_components=0.95, method="fastica", verbose=False)
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

        # from scipy.signal import welch
        # muscle_candidates = []
        # for ic_idx in range(n_ic):
        #     f, Pxx = welch(sources[ic_idx], sfreq, nperseg=512)
        #     hf_mask = (f >= 20)
        #     hf_ratio = Pxx[hf_mask].sum() / np.sum(Pxx)
        #     if hf_ratio > 0.25:  # threshold – tune as needed
        #         muscle_candidates.append(ic_idx)

        # exclude = list(set(eog_inds + muscle_candidates))
        exclude = list(set(eog_inds))
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

    def classify(self, df:pd.DataFrame) -> tuple[str, float]:
        """Classify EEG data for ADHD probability"""
        logger.info(f"Starting EEG classification on {len(df)} samples")
        window_count = 0
        n_samples = len(df)
        output = []

        # From now on, the subset_df is just the electrodes.
        df = df[ELECTRODE_CHANNELS]
        df = self.apply_filter(df, 0.5, 30)
        df = self.apply_ica_cleaning_to_dataset(df)

        n_samples = len(df)

        # Initialize power accumulators
        powers = {band: 0 for band in FREQUENCY_BANDS.keys()}

        # Compute band powers for each channel and average across channels
        for channel in ELECTRODE_CHANNELS:
            if channel in df.columns:
                eeg_data = df[channel].values

                for band_name, band_range in FREQUENCY_BANDS.items():
                    powers[band_name] += self.compute_band_power(eeg_data, SAMPLE_RATE, band_range)

        # Average across channels
        n_channels = len(ELECTRODE_CHANNELS)
        for band_name in powers.keys():
            powers[band_name] /= n_channels

        # Compute ratios
        tbr = powers['theta'] / powers['beta'] if powers['beta'] != 0 else np.nan
        tar = powers['theta'] / powers['alpha'] if powers['alpha'] != 0 else np.nan

        # Store power results
        power_results = {
            'theta': powers['theta'],
            'beta': powers['beta'],
            'alpha': powers['alpha'],
            'fast_alpha': powers['fast_alpha'],
            'high_beta': powers['high_beta'],
            'tbr': tbr,
            'tar': tar,
        }

        std_dev = df.std()
        mean = df.mean()
        df = (df - mean) / std_dev

        # Sliding window with overlap
        # Takes WINDOW_STEP steps every iteration
        for start in range(0, n_samples - SAMPLES_PER_WINDOW + 1, WINDOW_STEP):
            window = df.iloc[start:start + SAMPLES_PER_WINDOW]
            for frequency in self.process_window(window, window_count):
                output.append(frequency)
            window_count += 1

        logger.info(f"Processed {window_count} windows from EEG data")

        # Create frequency dataframe and add power columns
        frequency_df = pd.DataFrame(output)
        for power_name, power_value in power_results.items():
            frequency_df[power_name] = power_value

        frequency_count = len(frequency_df['Frequency'].unique())
        window_count = len(frequency_df['Window'].unique())
        numeric_df = frequency_df.drop(['Window','Frequency'], axis=1)

        # shape: (windows, freq, features)
        # Reshaping dataframe into tensor for classification
        logger.debug(f"Reshaping data: {window_count} windows, {frequency_count} frequencies, {numeric_df.shape[1]} features")
        full_ndarray = numeric_df.values.reshape((window_count, frequency_count, numeric_df.shape[1]))
        full_ndarray = full_ndarray[..., np.newaxis]

        logger.info("Running model inference")
        with torch.no_grad():
            tensor = torch.tensor(full_ndarray, dtype=torch.float32).permute(0,3,1,2)

            print(f"The tensor shape: {tensor.shape}")
            x_eeg  = tensor[:, :, :, :19]    # (1, 77, 19)
            x_band = tensor[:, 0, 0, 19:]    # (7,)

            print((x_eeg.shape, x_band.shape))
            predictions = self.model_loader.model(x_eeg, x_band).softmax(1).detach().numpy()
            print("="*100)
            print(f"Predictions: {predictions}")
            print("="*100)

            adhd_1_name = "ADHD 1"
            adhd_2_name = "ADHD 2"
            adhd_3_name = "ADHD 3"
            control_name = "Control"

            adhd_1, adhd_2, adhd_3, control = np.sum(predictions, axis=0) / np.sum(predictions)
            maximum = max(adhd_1, adhd_2, adhd_3, control)
            conf = float(maximum * 100)

            if -0.001 <= maximum - adhd_1 <= 0.001:
                logger.info(f"Classification result: {adhd_1_name} with {conf:.2f}% confidence")
                return adhd_1_name, conf
            elif -0.001 <= maximum - adhd_2 <= 0.001:
                logger.info(f"Classification result: {adhd_2_name} with {conf:.2f}% confidence")
                return adhd_2_name, conf
            elif -0.001 <= maximum - adhd_3 <= 0.001:
                logger.info(f"Classification result: {adhd_3_name} with {conf:.2f}% confidence")
                return adhd_3_name, conf
            elif -0.001 <= maximum - control <= 0.001:
                logger.info(f"Classification result: {control_name} with {conf:.2f}% confidence")
                return control_name, conf


    def classify_and_save(self, recording_id:int, df:pd.DataFrame)->dict:
        """Classify EEG data and save results to database."""
        logger.info(f"Starting classify and save for recording {recording_id}")
        classification, confidence = self.classify(df)

        logger.info(f"Saving classification result to database: {classification} ({confidence:.4f})")
        # result_id = self.results_repo.create_result(
        #     recording_id=recording_id,
        #     classification=classification,
        #     confidence_score=confidence
        # )

        # logger.info(f"Classification complete for recording {recording_id}, result ID: {result_id}")
        # return {
        #     'recording_id': recording_id,
        #     'result_id': result_id,
        #     'classification': classification,
        #     'confidence_score': confidence
        # }

        logger.info(f"Classification complete for recording {recording_id}, result ID: {None}")
        return {
            'recording_id': recording_id,
            'result_id': None,
            'classification': classification,
            'confidence_score': confidence
        }
