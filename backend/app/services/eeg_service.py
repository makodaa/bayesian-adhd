import torch
import numpy as np
import pandas as pd
from ..ml.model_loader import ModelLoader
from ..config import SAMPLE_RATE, SAMPLES_PER_WINDOW, WINDOW_STEP, TARGET_FREQUENCY_BINS
from ..db.repositories.results import ResultsRepository
from ..db.repositories.recordings import RecordingsRepository
from ..core.logging_config import get_app_logger

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

    def process_window(self,window:pd.DataFrame, window_count:int) -> list[dict]:
        """Process a single window and compute frequency features."""
        logger.debug(f"Processing window {window_count} with {len(window)} samples")
        output = []
        electrode_columns = window.select_dtypes(include=[np.number]).columns
        n = len(window)
        original_freqs = np.fft.rfftfreq(n,d=1/SAMPLE_RATE)

        # compute power spectra for all electrodes
        electrode_powers: dict[str, np.ndarray] = {}
        for electrode in electrode_columns:
            signal = window[electrode].to_numpy()
            fft_vals = np.fft.rfft(signal)
            power = np.abs(fft_vals) ** 2

            # interpolate to common freq bins
            electrode_powers[electrode] = np.interp(TARGET_FREQUENCY_BINS, original_freqs, power)

        # build rows: one per frequency bin
        for i, f in enumerate(TARGET_FREQUENCY_BINS):
            output.append({
                "window":window_count,
                "frequency":f,
                **{electrode:electrode_powers[electrode][i] for electrode in electrode_columns}
            })

        return output
        
    def classify(self, df:pd.DataFrame)->float:
        """Classify EEG data for ADHD probability"""
        logger.info(f"Starting EEG classification on {len(df)} samples")
        window_count = 0
        n_samples = len(df)
        output = []

        # Sliding window with overlap
        # Takes WINDOW_STEP steps every iteration
        for start in range(0, n_samples - SAMPLES_PER_WINDOW + 1, WINDOW_STEP):
            window = df.iloc[start:start + SAMPLES_PER_WINDOW]
            for frequency in self.process_window(window, window_count):
                output.append(frequency)
            window_count += 1

        logger.info(f"Processed {window_count} windows from EEG data")
        windows_dataframe = pd.DataFrame(output)
        frequency_count = len(windows_dataframe['frequency'].unique())
        window_count = len(windows_dataframe['window'].unique())
        numeric_df = windows_dataframe.drop(['window','frequency'], axis=1)

        # shape: (windows, freq, features)
        # Reshaping dataframe into tensor for classification
        logger.debug(f"Reshaping data: {window_count} windows, {frequency_count} frequencies, {numeric_df.shape[1]} features")
        full_ndarray = numeric_df.to_numpy().reshape((window_count, frequency_count, numeric_df.shape[1]))
        full_ndarray = self.model_loader.scaler.transform(full_ndarray)
        full_ndarray = full_ndarray[..., np.newaxis]

        logger.info("Running model inference")
        with torch.no_grad():
            tensor = torch.tensor(full_ndarray, dtype=torch.float32).permute(0,3,1,2)
            predictions = self.model_loader.model(tensor).softmax(1).detach().numpy()
            adhd, control = np.sum(predictions, axis=0) / np.sum(predictions)

            if adhd > control:
                logger.info(f"Classification result: ADHD with {adhd * 100:.2f}% confidence")
                return float(adhd)
            else:
                logger.info(f"Classification result: Control with {control * 100:.2f}% confidence")
                return float(-control)
            
    def classify_and_save(self, recording_id:int, df:pd.DataFrame)->dict:
        """Classify EEG data and save results to database."""
        logger.info(f"Starting classify and save for recording {recording_id}")
        result_value:float = self.classify(df)

        classification = 'ADHD' if result_value > 0 else 'Control'
        confidence = abs(result_value)
        
        logger.info(f"Saving classification result to database: {classification} ({confidence:.4f})")
        result_id = self.results_repo.create_result(
            recording_id = recording_id,
            classification = classification,
            confidence_score = confidence
        )

        logger.info(f"Classification complete for recording {recording_id}, result ID: {result_id}")
        return {
            'recording_id': recording_id,
            'result_id': result_id,
            'classification': classification,
            'confidence_score': confidence
        }
