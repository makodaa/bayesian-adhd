import numpy as np
import pandas as pd
from scipy.signal import welch
from ..config import SAMPLE_RATE, FREQUENCY_BANDS
from ..db.repositories.band_powers import BandPowersRepository
from ..db.repositories.ratios import RatiosRepository
from ..core.logging_config import get_app_logger

logger = get_app_logger(__name__)

class BandAnalysisService:
    """
    Compute absolute and relative power for each frequency band.
    
    Frequency bands:
    - Delta: 0.5-4 Hz
    - Theta: 4-8 Hz
    - Alpha: 8-13 Hz
    - Beta: 13-30 Hz
    - Gamma: 30-60 Hz
    
    Returns:
    --------
    dict : Contains absolute and relative power for each band and electrode
    """
    
    def __init__(self,band_powers_repo:BandPowersRepository,ratios_repo:RatiosRepository):
        self.band_powers_repo = band_powers_repo
        self.ratios_repo = ratios_repo
    
    def compute_band_powers(self, df: pd.DataFrame) -> dict:
        """Compute absolute and relative power for each frequency band"""
        logger.info(f"Computing band powers for {len(df)} samples")
        electrodes = df.select_dtypes(include=[np.number]).columns.tolist()
        logger.debug(f"Processing {len(electrodes)} electrodes: {electrodes}")
        result = {
            'absolute_power' : {},
            'relative_power' : {},
            'total_power' : {},
            'band_ratios' : {}
        }

        for electrode in electrodes:
            signal = df[electrode].to_numpy()

            # Compute power spectral density (PSD) using Welch's method
            freqs, psd = welch(signal, fs=SAMPLE_RATE, nperseg=min(256, len(signal)//4))

            # Compute absolute power for each band
            absolute_powers = {}
            for band_name, (low,high) in FREQUENCY_BANDS.items():
                band_mask = (freqs >= low) & (freqs < high)
                band_power = np.trapezoid(psd[band_mask], freqs[band_mask])
                absolute_powers[band_name] = float(band_power)

            # Compute total power across all bands
            total_power = sum(absolute_powers.values())

            # Compute relative power (as fraction of total)
            relative_powers = {
                band_name: (power / total_power) if total_power > 0 else 0.0
                for band_name, power in absolute_powers.items()
            }

            # Store results for this electrode
            result['absolute_power'][electrode] = absolute_powers
            result['relative_power'][electrode] = relative_powers
            result['total_power'][electrode] = float(total_power)

        result['average_absolute_power'] = {
            band: np.mean([result['absolute_power'][elec][band] for elec in electrodes])
            for band in FREQUENCY_BANDS.keys()
        }

        result['average_relative_power'] = {
            band: np.mean([result['relative_power'][elec][band] for elec in electrodes])
            for band in FREQUENCY_BANDS.keys()
        }

        avg_theta = result['average_absolute_power']['theta']
        avg_beta = result['average_absolute_power']['beta']
        avg_alpha = result['average_absolute_power']['alpha']

        result['band_ratios'] = {
            'theta_beta_ratio': float(avg_theta/avg_beta) if avg_beta > 0 else 0.0,
            'theta_alpha_ratio': float(avg_theta/avg_alpha) if avg_alpha > 0 else 0.0,
            'alpha_theta_ratio': float(avg_alpha/avg_theta) if avg_theta > 0 else 0.0,
        }

        logger.info(f"Band power computation complete. Ratios: theta/beta={result['band_ratios']['theta_beta_ratio']:.4f}")

        return result

    def compute_and_save(self, result_id:int, df:pd.DataFrame) -> dict:
        """Compute band powers and save to database."""
        logger.info(f"Computing and saving band powers for result {result_id}")
        powers = self.compute_band_powers(df)

        # Save band powers for each electrode
        band_power_count = 0
        for electrode, absolute_powers in powers['absolute_power'].items():
            relative_powers = powers['relative_power'][electrode]
            for band, power in absolute_powers.items():
                self.band_powers_repo.create_band_power(
                    result_id=result_id,
                    electrode=electrode,
                    frequency_band=band,
                    absolute_power=power,
                    relative_power=relative_powers[band]
                )
                band_power_count += 1

        logger.info(f"Saved {band_power_count} band power entries for result {result_id}")

        # Save ratios
        for ratio, value in powers['band_ratios'].items():
            self.ratios_repo.create_ratio(
                result_id=result_id,
                ratio_name=ratio,
                ratio_value=value
            )

        logger.info(f"Saved {len(powers['band_ratios'])} ratio entries for result {result_id}")

        return powers