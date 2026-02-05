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

    def classify(self, df:pd.DataFrame) -> tuple[str, float, dict]:
        """Classify EEG data for ADHD probability"""
        logger.info(f"Starting EEG classification on {len(df)} samples")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")

        # Check if columns are numeric (0-18 or 1-19) and rename them to electrode names
        if list(df.columns) == [str(i) for i in range(19)]:
            logger.info("Detected numeric column names (0-18), renaming to electrode channels")
            df.columns = ELECTRODE_CHANNELS
        elif list(df.columns) == list(range(19)):
            logger.info("Detected numeric column names (0-18), renaming to electrode channels")
            df.columns = ELECTRODE_CHANNELS
        elif list(df.columns) == [str(i) for i in range(1, 20)]:
            logger.info("Detected numeric column names (1-19), renaming to electrode channels")
            df.columns = ELECTRODE_CHANNELS
        elif list(df.columns) == list(range(1, 20)):
            logger.info("Detected numeric column names (1-19), renaming to electrode channels")
            df.columns = ELECTRODE_CHANNELS
        
        # Check if expected columns exist
        missing_channels = [ch for ch in ELECTRODE_CHANNELS if ch not in df.columns]
        if missing_channels:
            logger.error(f"Missing electrode channels: {missing_channels}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"CSV file is missing required electrode channels: {missing_channels}. "
                            f"Expected channels: {ELECTRODE_CHANNELS}. "
                            f"Found columns: {df.columns.tolist()}")
        window_count = 0
        n_samples = len(df)
        output = []

        # From now on, the subset_df is just the electrodes.
        df = df[ELECTRODE_CHANNELS]
        df = self.apply_filter(df, 0.5, 60)
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

        # Compute total power and relative powers for frontend display
        total_power = sum(powers.values())
        relative_powers = {
            band: float(power / total_power) if total_power > 0 else 0.0
            for band, power in powers.items()
        }

        # Standard bands for frontend display (exclude fast_alpha and high_beta)
        DISPLAY_BANDS = {'delta', 'theta', 'alpha', 'beta', 'gamma'}
        
        # Prepare band data for return (matches frontend expectations)
        # Full data includes all bands for database storage
        band_data = {
            'average_absolute_power': {band: float(power) for band, power in powers.items()},
            'average_relative_power': relative_powers,
            'band_ratios': {
                'theta_beta_ratio': float(tbr) if not np.isnan(tbr) else 0.0,
                'theta_alpha_ratio': float(tar) if not np.isnan(tar) else 0.0,
            }
        }
        
        # Filtered data for frontend display (only standard 5 bands)
        band_data['display_bands'] = {
            'average_absolute_power': {band: power for band, power in band_data['average_absolute_power'].items() if band in DISPLAY_BANDS},
            'average_relative_power': {band: power for band, power in band_data['average_relative_power'].items() if band in DISPLAY_BANDS},
            'band_ratios': band_data['band_ratios']
        }

        # Store power results for model features
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

            adhd_1_name = "Combined / C (ADHD-C)"
            adhd_2_name = "Hyperactive-Impulsive (ADHD-HI)"
            adhd_3_name = "Inattentive (ADHD-I)"
            control_name = "Non-ADHD"

            adhd_1, adhd_2, adhd_3, control = np.sum(predictions, axis=0) / np.sum(predictions)
            maximum = max(adhd_1, adhd_2, adhd_3, control)
            # Store confidence as decimal (0-1) - frontend/PDF will multiply by 100 for display
            conf = float(maximum)

            if -0.001 <= maximum - adhd_1 <= 0.001:
                logger.info(f"Classification result: {adhd_1_name} with {conf*100:.2f}% confidence")
                return adhd_1_name, conf, band_data
            elif -0.001 <= maximum - adhd_2 <= 0.001:
                logger.info(f"Classification result: {adhd_2_name} with {conf*100:.2f}% confidence")
                return adhd_2_name, conf, band_data
            elif -0.001 <= maximum - adhd_3 <= 0.001:
                logger.info(f"Classification result: {adhd_3_name} with {conf*100:.2f}% confidence")
                return adhd_3_name, conf, band_data
            elif -0.001 <= maximum - control <= 0.001:
                logger.info(f"Classification result: {control_name} with {conf*100:.2f}% confidence")
                return control_name, conf, band_data


    def classify_and_save(self, recording_id:int, df:pd.DataFrame, clinician_id:int=None)->dict:
        """Classify EEG data and save results to database."""
        logger.info(f"Starting classify and save for recording {recording_id}")
        classification, confidence, band_data = self.classify(df)

        logger.info(f"Saving classification result to database: {classification} ({confidence*100:.2f}%)")
        result_id = self.results_repo.create_result(
            recording_id = recording_id,
            classification = classification,
            confidence_score = confidence,
            clinician_id = clinician_id
        )

        logger.info(f"Classification complete for recording {recording_id}, result ID: {result_id}")
        return {
            'recording_id': recording_id,
            'result_id': result_id,
            'classification': classification,
            'confidence_score': confidence,
            'clinician_id': clinician_id
        }
