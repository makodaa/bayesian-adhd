"""
Temporal Importance Service

This service implements sliding window occlusion sensitivity analysis to determine which
time segments contribute most to the ADHD classification decision.

Method:
- Slide a temporal occlusion window across the EEG signal
- For each window position, mask the temporal segment
- Run inference with occluded window
- Measure probability drop from baseline
- Higher drop = more important time segment

Output:
- Time-importance curve showing when the model is most sensitive
- Line plot: X-axis = time, Y-axis = importance score
"""

import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional

from ..config import (
    CLASSIFYING_FREQUENCY_BANDS,
    ELECTRODE_CHANNELS,
    SAMPLE_RATE,
    SAMPLES_PER_WINDOW,
    TARGET_FREQUENCY_BINS,
    WINDOW_STEP,
)
from ..core.logging_config import get_app_logger
from ..ml.model_loader import ModelLoader
from ..db.repositories.temporal_importance import TemporalImportanceRepository

logger = get_app_logger(__name__)


class TemporalImportanceService:
    """Compute temporal importance for EEG classification using sliding window occlusion"""

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader

    def _prepare_tensor_data(
        self, df: pd.DataFrame, apply_preprocessing: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare EEG data for model inference.
        
        Args:
            df: DataFrame with electrode channels as columns
            apply_preprocessing: Whether to apply normalization and windowing
            
        Returns:
            Tuple of (x_eeg, x_band) tensors ready for model
        """
        from ..services.eeg_service import EEGService
        
        # Use temporary EEG service for preprocessing methods
        temp_service = EEGService(self.model_loader, None, None)
        
        # Apply preprocessing if needed
        if apply_preprocessing:
            df = df[ELECTRODE_CHANNELS].copy()
            df = temp_service.apply_filter(df, 0.5, 60)
            df = temp_service.apply_ica_cleaning_to_dataset(df)
            
        # Compute band powers
        powers = {band: 0.0 for band in CLASSIFYING_FREQUENCY_BANDS.keys()}
        
        for channel in ELECTRODE_CHANNELS:
            if channel in df.columns:
                eeg_data = df[channel].values
                for band_name, band_range in CLASSIFYING_FREQUENCY_BANDS.items():
                    powers[band_name] += temp_service.compute_band_power(
                        eeg_data, SAMPLE_RATE, band_range
                    )
        
        # Average across channels
        n_channels = len(ELECTRODE_CHANNELS)
        for band_name in powers.keys():
            powers[band_name] /= n_channels
            if np.isnan(powers[band_name]) or np.isinf(powers[band_name]):
                powers[band_name] = 0.0
            
        # Compute ratios with NaN protection
        tbr = powers["theta"] / powers["beta"] if powers["beta"] != 0 and not np.isnan(powers["beta"]) else 0.0
        tar = powers["theta"] / powers["alpha"] if powers["alpha"] != 0 and not np.isnan(powers["alpha"]) else 0.0
        
        if np.isnan(tbr) or np.isinf(tbr):
            tbr = 0.0
        if np.isnan(tar) or np.isinf(tar):
            tar = 0.0
        
        power_results = {
            "theta": powers["theta"],
            "beta": powers["beta"],
            "alpha": powers["alpha"],
            "fast_alpha": powers["fast_alpha"],
            "high_beta": powers["high_beta"],
            "tbr": tbr,
            "tar": tar,
        }
        
        # Normalize the data
        std_dev = df.std()
        mean = df.mean()
        std_dev = std_dev.replace(0, 1)
        df = (df - mean) / std_dev
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Process windows
        n_samples = len(df)
        output = []
        window_count = 0
        
        for start in range(0, n_samples - SAMPLES_PER_WINDOW + 1, WINDOW_STEP):
            window = df.iloc[start : start + SAMPLES_PER_WINDOW]
            for frequency in temp_service.process_window(window, window_count):
                output.append(frequency)
            window_count += 1
            
        # Create frequency dataframe
        frequency_df = pd.DataFrame(output)
        for power_name, power_value in power_results.items():
            frequency_df[power_name] = power_value
            
        frequency_count = len(frequency_df["Frequency"].unique())
        window_count = len(frequency_df["Window"].unique())
        numeric_df = frequency_df.drop(["Window", "Frequency"], axis=1)
        
        # Reshape to tensor format
        full_ndarray = numeric_df.values.reshape(
            (window_count, frequency_count, numeric_df.shape[1])
        )
        full_ndarray = full_ndarray[..., np.newaxis]
        
        # Convert to tensor
        tensor = torch.tensor(full_ndarray, dtype=torch.float32).permute(0, 3, 1, 2)
        x_eeg = tensor[:, :, :, :19]  # (batch, 1, freq_bins, 19 channels)
        x_band = tensor[:, 0, 0, 19:]  # (batch, 7 band features)
        
        return x_eeg, x_band

    def _run_inference(
        self, x_eeg: torch.Tensor, x_band: torch.Tensor
    ) -> np.ndarray:
        """
        Run model inference and return probabilities.
        
        Args:
            x_eeg: EEG tensor (batch, 1, freq_bins, channels)
            x_band: Band features tensor (batch, 7)
            
        Returns:
            Array of class probabilities (batch, 4)
        """
        with torch.no_grad():
            logits = self.model_loader.model(x_eeg, x_band)
            probabilities = logits.softmax(1).detach().numpy()
        return probabilities

    def compute_temporal_importance(
        self,
        df: pd.DataFrame,
        window_size_ms: int = 500,
        stride_ms: int = 100,
        target_class_idx: int = None
    ) -> Dict:
        """
        Compute temporal importance using sliding window occlusion.
        
        For each time window:
        1. Run baseline inference with all data intact
        2. Mask the temporal window (set to zero)
        3. Run inference again
        4. Importance = baseline_prob - occluded_prob
        
        Args:
            df: DataFrame with EEG data (electrode columns)
            window_size_ms: Size of occlusion window in milliseconds (default: 500ms)
            stride_ms: Stride for sliding window in milliseconds (default: 100ms)
            target_class_idx: Class to compute importance for (0-3).
                             If None, uses the predicted class.
            
        Returns:
            Dictionary containing:
            - time_points: List of time points (in seconds)
            - importance_scores: List of importance scores at each time point
            - baseline_probability: Baseline prediction probability
            - predicted_class_idx: Predicted class index
            - window_size_ms: Window size used
            - stride_ms: Stride used
            - total_duration_sec: Total duration of signal
        """
        logger.info(f"Computing temporal importance with window_size={window_size_ms}ms, stride={stride_ms}ms")
        
        # Validate columns
        missing_channels = [ch for ch in ELECTRODE_CHANNELS if ch not in df.columns]
        if missing_channels:
            raise ValueError(
                f"Missing required channels: {missing_channels}. "
                f"Expected: {ELECTRODE_CHANNELS}"
            )
        
        # Convert window size and stride from ms to samples
        window_size_samples = int((window_size_ms / 1000.0) * SAMPLE_RATE)
        stride_samples = int((stride_ms / 1000.0) * SAMPLE_RATE)
        
        total_samples = len(df)
        total_duration_sec = total_samples / SAMPLE_RATE
        
        logger.info(f"Signal: {total_samples} samples, {total_duration_sec:.2f} seconds")
        logger.info(f"Occlusion window: {window_size_samples} samples ({window_size_ms}ms)")
        logger.info(f"Stride: {stride_samples} samples ({stride_ms}ms)")
        
        # Get baseline prediction
        logger.debug("Running baseline inference")
        x_eeg_baseline, x_band_baseline = self._prepare_tensor_data(df)
        baseline_probs = self._run_inference(x_eeg_baseline, x_band_baseline)
        baseline_mean_probs = np.mean(baseline_probs, axis=0)
        
        # Determine target class
        if target_class_idx is None:
            target_class_idx = int(np.argmax(baseline_mean_probs))
        
        baseline_target_prob = baseline_mean_probs[target_class_idx]
        logger.info(f"Baseline probability for class {target_class_idx}: {baseline_target_prob:.4f}")
        
        # Compute importance for each time window
        time_points = []
        importance_scores = []
        
        num_windows = (total_samples - window_size_samples) // stride_samples + 1
        logger.info(f"Processing {num_windows} temporal windows")
        
        for i in range(num_windows):
            start_idx = i * stride_samples
            end_idx = start_idx + window_size_samples
            
            # Ensure we don't exceed bounds
            if end_idx > total_samples:
                break
            
            # Calculate center time point for this window
            center_idx = (start_idx + end_idx) // 2
            time_sec = center_idx / SAMPLE_RATE
            
            # Create occluded version with this window masked
            df_occluded = df.copy()
            for channel in ELECTRODE_CHANNELS:
                df_occluded.iloc[start_idx:end_idx, df_occluded.columns.get_loc(channel)] = 0.0
            
            # Run inference with occluded window
            x_eeg_occluded, x_band_occluded = self._prepare_tensor_data(df_occluded)
            occluded_probs = self._run_inference(x_eeg_occluded, x_band_occluded)
            occluded_mean_probs = np.mean(occluded_probs, axis=0)
            occluded_target_prob = occluded_mean_probs[target_class_idx]
            
            # Importance = drop in probability when window is masked
            importance = baseline_target_prob - occluded_target_prob
            
            # Handle NaN or inf values
            if np.isnan(importance) or np.isinf(importance):
                logger.warning(f"Window at {time_sec:.2f}s has invalid importance, setting to 0")
                importance = 0.0
            
            time_points.append(float(time_sec))
            importance_scores.append(float(importance))
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i+1}/{num_windows} windows")
        
        logger.info("Temporal importance computation complete")
        
        return {
            "time_points": time_points,
            "importance_scores": importance_scores,
            "baseline_probability": float(baseline_target_prob),
            "predicted_class_idx": int(target_class_idx),
            "window_size_ms": window_size_ms,
            "stride_ms": stride_ms,
            "total_duration_sec": float(total_duration_sec),
            "num_windows": len(time_points)
        }

    def compute_temporal_importance_with_classification(
        self,
        df: pd.DataFrame,
        window_size_ms: int = 500,
        stride_ms: int = 100
    ) -> Dict:
        """
        Compute temporal importance along with classification results.
        
        Args:
            df: DataFrame with EEG data
            window_size_ms: Size of occlusion window in milliseconds
            stride_ms: Stride for sliding window in milliseconds
            
        Returns:
            Dictionary containing classification and temporal importance data
        """
        logger.info("Computing temporal importance with classification")
        
        # Get baseline prediction for classification
        x_eeg_baseline, x_band_baseline = self._prepare_tensor_data(df)
        baseline_probs = self._run_inference(x_eeg_baseline, x_band_baseline)
        baseline_mean_probs = np.mean(baseline_probs, axis=0)
        
        # Determine classification
        class_names = [
            "Combined / C (ADHD-C)",
            "Hyperactive-Impulsive (ADHD-H)",
            "Inattentive (ADHD-I)",
            "Non-ADHD"
        ]
        
        predicted_class_idx = int(np.argmax(baseline_mean_probs))
        predicted_class = class_names[predicted_class_idx]
        confidence = float(baseline_mean_probs[predicted_class_idx])
        
        class_probabilities = {
            class_names[i]: float(baseline_mean_probs[i])
            for i in range(len(class_names))
        }
        
        # Compute temporal importance
        temporal_data = self.compute_temporal_importance(
            df,
            window_size_ms=window_size_ms,
            stride_ms=stride_ms,
            target_class_idx=predicted_class_idx
        )
        
        # Combine results
        result = {
            "classification": predicted_class,
            "confidence": confidence,
            "predicted_class_idx": predicted_class_idx,
            "class_probabilities": class_probabilities,
            "temporal_importance": temporal_data
        }
        
        return result

    def compute_temporal_importance_with_visualizations(
        self,
        df: pd.DataFrame,
        window_size_ms: int = 500,
        stride_ms: int = 100,
        result_id: int = None
    ) -> Dict:
        """
        Compute temporal importance with visualizations and optionally save to database.
        
        Args:
            df: DataFrame with EEG data
            window_size_ms: Size of occlusion window in milliseconds
            stride_ms: Stride for sliding window in milliseconds
            result_id: Optional result ID to save to database
            
        Returns:
            Dictionary with classification, temporal importance, and visualizations
        """
        logger.info("Computing temporal importance with visualizations")
        
        # Compute temporal importance with classification
        result = self.compute_temporal_importance_with_classification(
            df,
            window_size_ms=window_size_ms,
            stride_ms=stride_ms
        )
        
        # Generate visualizations
        temporal_data = result["temporal_importance"]
        time_points = temporal_data["time_points"]
        importance_scores = temporal_data["importance_scores"]
        
        # 1. Time-importance curve
        time_curve_img = self._create_time_importance_plot(
            time_points,
            importance_scores,
            result["classification"],
            result["confidence"]
        )
        
        # 2. Heatmap view of importance over time
        heatmap_img = self._create_temporal_heatmap(
            time_points,
            importance_scores
        )
        
        # 3. Statistics plot
        stats_img = self._create_temporal_statistics_plot(
            time_points,
            importance_scores,
            temporal_data["baseline_probability"]
        )
        
        result["visualizations"] = {
            "time_curve": time_curve_img,
            "heatmap": heatmap_img,
            "statistics": stats_img
        }
        
        # Save to database if result_id provided
        if result_id is not None:
            try:
                TemporalImportanceRepository.save(
                    result_id=result_id,
                    predicted_class=result["classification"],
                    confidence_score=result["confidence"],
                    time_points=time_points,
                    importance_scores=importance_scores,
                    window_size_ms=window_size_ms,
                    stride_ms=stride_ms,
                    time_curve_plot=time_curve_img,
                    heatmap_plot=heatmap_img,
                    statistics_plot=stats_img
                )
                logger.info(f"Saved temporal importance to database for result_id={result_id}")
            except Exception as e:
                logger.error(f"Failed to save temporal importance to database: {e}", exc_info=True)
        
        return result

    def _create_time_importance_plot(
        self,
        time_points: List[float],
        importance_scores: List[float],
        classification: str,
        confidence: float
    ) -> str:
        """Create line plot of importance over time."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot importance curve
        ax.plot(time_points, importance_scores, linewidth=2, color='#2E86AB', marker='o', markersize=4)
        ax.fill_between(time_points, importance_scores, alpha=0.3, color='#2E86AB')
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Highlight most important regions
        if importance_scores:
            mean_importance = np.mean(importance_scores)
            std_importance = np.std(importance_scores)
            threshold = mean_importance + std_importance
            
            for i, (t, imp) in enumerate(zip(time_points, importance_scores)):
                if imp > threshold:
                    ax.axvspan(
                        t - 0.1 if i > 0 else t,
                        t + 0.1 if i < len(time_points) - 1 else t,
                        alpha=0.2,
                        color='red'
                    )
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Importance Score', fontsize=12)
        ax.set_title(
            f'Temporal Importance Analysis\n{classification} (Confidence: {confidence:.2%})',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_str}"

    def _create_temporal_heatmap(
        self,
        time_points: List[float],
        importance_scores: List[float]
    ) -> str:
        """Create heatmap visualization of temporal importance."""
        fig, ax = plt.subplots(figsize=(12, 3))
        
        # Reshape for heatmap (1 row)
        data = np.array(importance_scores).reshape(1, -1)
        
        im = ax.imshow(
            data,
            aspect='auto',
            cmap='RdYlBu_r',
            interpolation='bilinear'
        )
        
        # Set x-axis to show time
        num_ticks = min(10, len(time_points))
        tick_indices = np.linspace(0, len(time_points) - 1, num_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f"{time_points[i]:.1f}s" for i in tick_indices])
        
        ax.set_yticks([])
        ax.set_xlabel('Time', fontsize=12)
        ax.set_title('Temporal Importance Heatmap', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15)
        cbar.set_label('Importance Score', fontsize=10)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_str}"

    def _create_temporal_statistics_plot(
        self,
        time_points: List[float],
        importance_scores: List[float],
        baseline_prob: float
    ) -> str:
        """Create statistical summary plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Distribution of importance scores
        ax1.hist(importance_scores, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(importance_scores), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(importance_scores):.4f}')
        ax1.axvline(np.median(importance_scores), color='green', linestyle='--',
                    label=f'Median: {np.median(importance_scores):.4f}')
        ax1.set_xlabel('Importance Score', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Distribution of Temporal Importance', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right: Summary statistics
        stats_text = [
            f"Total Duration: {time_points[-1]:.2f} seconds" if time_points else "N/A",
            f"Baseline Probability: {baseline_prob:.4f}",
            f"Mean Importance: {np.mean(importance_scores):.4f}",
            f"Std Importance: {np.std(importance_scores):.4f}",
            f"Max Importance: {np.max(importance_scores):.4f}",
            f"Min Importance: {np.min(importance_scores):.4f}",
            f"Number of Windows: {len(importance_scores)}"
        ]
        
        # Find peak importance times
        if importance_scores:
            sorted_indices = np.argsort(importance_scores)[::-1][:3]
            stats_text.append("\nTop 3 Most Important Times:")
            for idx in sorted_indices:
                stats_text.append(f"  {time_points[idx]:.2f}s: {importance_scores[idx]:.4f}")
        
        ax2.axis('off')
        ax2.text(0.1, 0.5, '\n'.join(stats_text), fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax2.set_title('Summary Statistics', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_str}"
