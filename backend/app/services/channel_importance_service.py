"""
Channel Importance Service

This service implements occlusion sensitivity analysis to determine which EEG channels
(brain regions) contribute most to the ADHD classification decision.

Method:
- For each channel, temporarily zero it out
- Run inference with occluded channel
- Measure probability drop from baseline
- Higher drop = more important channel
"""

import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional

from ..config import (
    CLASSIFYING_FREQUENCY_BANDS,
    ELECTRODE_CHANNELS,
    ELECTRODE_POSITIONS,
    SAMPLE_RATE,
    SAMPLES_PER_WINDOW,
    TARGET_FREQUENCY_BINS,
    WINDOW_STEP,
)
from ..core.logging_config import get_app_logger
from ..ml.model_loader import ModelLoader
from ..db.repositories.channel_importance import ChannelImportanceRepository

logger = get_app_logger(__name__)


class ChannelImportanceService:
    """Compute channel-wise importance for EEG classification using occlusion sensitivity"""

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
        # Note: We don't need the full service, just the preprocessing functions
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
            # Ensure no NaN values
            if np.isnan(powers[band_name]) or np.isinf(powers[band_name]):
                powers[band_name] = 0.0
            
        # Compute ratios with NaN protection
        tbr = powers["theta"] / powers["beta"] if powers["beta"] != 0 and not np.isnan(powers["beta"]) else 0.0
        tar = powers["theta"] / powers["alpha"] if powers["alpha"] != 0 and not np.isnan(powers["alpha"]) else 0.0
        
        # Ensure ratios are valid
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
        
        # Handle zero standard deviation (constant signals)
        std_dev = std_dev.replace(0, 1)  # Avoid division by zero
        
        df = (df - mean) / std_dev
        
        # Replace any NaN or inf values with 0
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

    def compute_channel_importance(
        self, df: pd.DataFrame, target_class_idx: int = None
    ) -> Dict[str, float]:
        """
        Compute importance score for each channel using occlusion sensitivity.
        
        For each channel:
        1. Run baseline inference with all channels
        2. Zero out the channel
        3. Run inference again
        4. Importance = baseline_prob - occluded_prob
        
        Args:
            df: DataFrame with EEG data (electrode columns)
            target_class_idx: Class to compute importance for (0-3).
                             If None, uses the predicted class.
            
        Returns:
            Dictionary mapping channel name to importance score
        """
        logger.info("Computing channel importance using occlusion sensitivity")
        
        # Validate columns
        missing_channels = [ch for ch in ELECTRODE_CHANNELS if ch not in df.columns]
        if missing_channels:
            raise ValueError(
                f"Missing required channels: {missing_channels}. "
                f"Expected: {ELECTRODE_CHANNELS}"
            )
        
        # Get baseline prediction
        logger.debug("Running baseline inference")
        x_eeg_baseline, x_band_baseline = self._prepare_tensor_data(df)
        baseline_probs = self._run_inference(x_eeg_baseline, x_band_baseline)
        
        # Average probabilities across all windows
        baseline_mean_probs = np.mean(baseline_probs, axis=0)
        
        # Determine target class (use predicted class if not specified)
        if target_class_idx is None:
            target_class_idx = int(np.argmax(baseline_mean_probs))
        
        baseline_target_prob = baseline_mean_probs[target_class_idx]
        logger.info(
            f"Baseline probability for class {target_class_idx}: {baseline_target_prob:.4f}"
        )
        
        # Compute importance for each channel
        importance_scores = {}
        
        for channel_idx, channel_name in enumerate(ELECTRODE_CHANNELS):
            logger.debug(f"Occluding channel {channel_name} (index {channel_idx})")
            
            # Create occluded version of data
            df_occluded = df.copy()
            df_occluded[channel_name] = 0.0
            
            # Run inference with occluded channel
            x_eeg_occluded, x_band_occluded = self._prepare_tensor_data(df_occluded)
            occluded_probs = self._run_inference(x_eeg_occluded, x_band_occluded)
            
            # Average probabilities across windows
            occluded_mean_probs = np.mean(occluded_probs, axis=0)
            occluded_target_prob = occluded_mean_probs[target_class_idx]
            
            # Importance = drop in probability when channel is removed
            importance = baseline_target_prob - occluded_target_prob
            
            # Handle NaN or inf values
            if np.isnan(importance) or np.isinf(importance):
                logger.warning(f"Channel {channel_name} has invalid importance: {importance}, setting to 0")
                importance = 0.0
            
            importance_scores[channel_name] = float(importance)
            
            logger.debug(
                f"Channel {channel_name}: "
                f"baseline={baseline_target_prob:.4f}, "
                f"occluded={occluded_target_prob:.4f}, "
                f"importance={importance:.4f}"
            )
        
        logger.info("Channel importance computation complete")
        return importance_scores

    def compute_channel_importance_with_classification(
        self, df: pd.DataFrame
    ) -> Dict:
        """
        Compute channel importance along with classification results.
        
        Args:
            df: DataFrame with EEG data
            
        Returns:
            Dictionary containing:
            - classification: Predicted class name
            - confidence: Confidence score
            - predicted_class_idx: Predicted class index (0-3)
            - class_probabilities: Dictionary of class probabilities
            - channel_importance: Dictionary of channel importance scores
            - importance_normalized: Normalized importance scores (0-1)
        """
        logger.info("Computing channel importance with classification")
        
        # Get baseline prediction
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
        
        logger.info(
            f"Classification: {predicted_class} with confidence {confidence:.4f}"
        )
        
        # Compute channel importance for predicted class
        importance_scores = self.compute_channel_importance(df, predicted_class_idx)
        
        # Clean importance scores of any NaN or inf values
        for ch, score in importance_scores.items():
            if np.isnan(score) or np.isinf(score):
                logger.warning(f"Channel {ch} has invalid score: {score}, setting to 0")
                importance_scores[ch] = 0.0
        
        # Normalize importance scores to [0, 1] range
        importance_values = np.array(list(importance_scores.values()))
        
        # Replace any remaining NaN or inf
        importance_values = np.nan_to_num(importance_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Handle case where all importances are the same or zero
        min_val = np.min(importance_values)
        max_val = np.max(importance_values)
        
        if np.isnan(min_val) or np.isnan(max_val) or max_val == min_val or (max_val - min_val) == 0:
            # All values are the same or invalid, use neutral value
            normalized_importance = {ch: 0.5 for ch in importance_scores.keys()}
        else:
            # Normalize to [0, 1]
            normalized_importance = {}
            for ch, score in importance_scores.items():
                norm_score = (score - min_val) / (max_val - min_val)
                # Final safety check
                if np.isnan(norm_score) or np.isinf(norm_score):
                    norm_score = 0.5
                normalized_importance[ch] = float(norm_score)
        
        return {
            "classification": predicted_class,
            "confidence": confidence,
            "predicted_class_idx": predicted_class_idx,
            "class_probabilities": class_probabilities,
            "channel_importance": importance_scores,
            "importance_normalized": normalized_importance,
        }

    def _create_mne_montage(self) -> mne.channels.DigMontage:
        """Create MNE montage from electrode positions."""
        # MNE expects positions in 3D, we'll set z=0 for 2D scalp plot
        positions_3d = {}
        for channel, (x, y) in ELECTRODE_POSITIONS.items():
            # Scale positions for MNE (expects roughly head size in meters)
            positions_3d[channel] = [x * 0.08, y * 0.08, 0.0]
        
        montage = mne.channels.make_dig_montage(
            ch_pos=positions_3d,
            coord_frame='head'
        )
        return montage

    def _plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"

    def _generate_fallback_topographic_map(self, importance_scores: Dict[str, float]) -> str:
        """
        Generate a simple scatter plot if MNE fails.
        
        Args:
            importance_scores: Dictionary mapping channel names to importance scores
            
        Returns:
            Base64-encoded PNG image
        """
        logger.info("Generating fallback topographic visualization")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Extract positions and values
        x_coords = []
        y_coords = []
        values = []
        labels = []
        
        for channel in ELECTRODE_CHANNELS:
            if channel in ELECTRODE_POSITIONS:
                pos = ELECTRODE_POSITIONS[channel]
                x_coords.append(pos[0])
                y_coords.append(pos[1])
                values.append(importance_scores[channel])
                labels.append(channel)
        
        # Create scatter plot with color mapping
        scatter = ax.scatter(x_coords, y_coords, c=values, cmap='hot', 
                           s=500, alpha=0.8, edgecolors='black', linewidth=2)
        
        # Add channel labels
        for x, y, label in zip(x_coords, y_coords, labels):
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white')
        
        # Draw head outline
        circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=3)
        ax.add_patch(circle)
        
        # Add nose
        nose_x = [0.1, 0, -0.1]
        nose_y = [1.0, 1.15, 1.0]
        ax.plot(nose_x, nose_y, 'k-', linewidth=3)
        
        # Formatting
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Channel Importance Topographic Map', fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Importance Score', rotation=270, labelpad=20)
        
        return self._plot_to_base64(fig)

    def generate_topographic_map(self, importance_scores: Dict[str, float]) -> str:
        """
        Generate topographic heatmap using MNE.
        
        Args:
            importance_scores: Dictionary mapping channel names to importance scores
            
        Returns:
            Base64-encoded PNG image
        """
        logger.info("Generating topographic map using MNE")
        
        try:
            # Create info object for MNE
            info = mne.create_info(
                ch_names=ELECTRODE_CHANNELS,
                sfreq=SAMPLE_RATE,
                ch_types='eeg'
            )
            
            # Set montage
            montage = self._create_mne_montage()
            info.set_montage(montage)
            
            # Prepare data for plotting (importance values in order of channels)
            data = np.array([importance_scores[ch] for ch in ELECTRODE_CHANNELS])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot topomap
            im, _ = mne.viz.plot_topomap(
                data,
                info,
                axes=ax,
                show=False,
                cmap='hot',
                contours=6,
                sensors=True,
                names=ELECTRODE_CHANNELS,
                # show_names=True,
                sphere='auto',
                res=64,
                outlines='head',
                image_interp='cubic'
            )
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Importance Score', rotation=270, labelpad=20)
            
            ax.set_title('Channel Importance Topographic Map', fontsize=14, fontweight='bold')
            
            return self._plot_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error generating MNE topomap: {e}", exc_info=True)
            logger.info("Falling back to scatter plot visualization")
            
            # Fallback: Create a simple scatter plot if MNE fails
            return self._generate_fallback_topographic_map(importance_scores)

    def generate_bar_chart(self, importance_scores: Dict[str, float]) -> str:
        """
        Generate bar chart of channel importance.
        
        Args:
            importance_scores: Dictionary mapping channel names to importance scores
            
        Returns:
            Base64-encoded PNG image
        """
        logger.info("Generating bar chart")
        
        # Sort by importance
        sorted_items = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        channels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(channels)))
        bars = ax.barh(channels, values, color=colors)
        
        # Formatting
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Channel', fontsize=12, fontweight='bold')
        ax.set_title('Channel Importance Scores (Sorted)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Highest values at top
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}',
                   ha='left', va='center', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        return self._plot_to_base64(fig)

    def generate_regional_chart(self, importance_scores: Dict[str, float]) -> str:
        """
        Generate regional analysis chart.
        
        Args:
            importance_scores: Dictionary mapping channel names to importance scores
            
        Returns:
            Base64-encoded PNG image
        """
        logger.info("Generating regional chart")
        
        # Define brain regions
        regions = {
            'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
            'Central': ['C3', 'C4', 'Cz'],
            'Temporal': ['T7', 'T8'],
            'Parietal': ['P3', 'P4', 'P7', 'P8', 'Pz'],
            'Occipital': ['O1', 'O2']
        }
        
        # Calculate average importance per region
        regional_importance = {}
        for region, channels in regions.items():
            values = [importance_scores[ch] for ch in channels if ch in importance_scores]
            if values:
                regional_importance[region] = np.mean(values)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        region_names = list(regional_importance.keys())
        region_values = list(regional_importance.values())
        
        # Create bar chart with gradient colors
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(region_names)))
        bars = ax.bar(region_names, region_values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Formatting
        ax.set_xlabel('Brain Region', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Average Importance by Brain Region', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return self._plot_to_base64(fig)

    def compute_channel_importance_with_visualizations(
        self, df: pd.DataFrame, result_id: Optional[int] = None
    ) -> Dict:
        """
        Compute channel importance and generate all visualizations.
        
        Args:
            df: DataFrame with EEG data
            result_id: Optional result ID to save channel importance to database
            
        Returns:
            Dictionary containing classification results, importance scores,
            and base64-encoded visualization images
        """
        logger.info("Computing channel importance with visualizations")
        
        # Get basic results
        result = self.compute_channel_importance_with_classification(df)
        
        # Generate visualizations
        logger.info("Generating visualizations")
        result["visualizations"] = {
            "topographic_map": self.generate_topographic_map(result["importance_normalized"]),
            "bar_chart": self.generate_bar_chart(result["channel_importance"]),
            "regional_chart": self.generate_regional_chart(result["channel_importance"])
        }
        
        # Save to database if result_id provided
        if result_id is not None:
            logger.info(f"Saving channel importance to database for result {result_id}")
            try:
                channel_importance_data = {
                    'topographic_map': result["visualizations"]["topographic_map"],
                    'bar_chart': result["visualizations"]["bar_chart"],
                    'regional_chart': result["visualizations"]["regional_chart"],
                    'importance_scores': result["channel_importance"],
                    'normalized_importance': result["importance_normalized"],
                    'predicted_class': result["classification"],
                    'confidence_score': result["confidence"]
                }
                
                channel_importance_id = ChannelImportanceRepository.save(result_id, channel_importance_data)
                result["channel_importance_id"] = channel_importance_id
                logger.info(f"Channel importance saved with ID {channel_importance_id}")
            except Exception as e:
                logger.error(f"Failed to save channel importance to database: {e}", exc_info=True)
                # Don't fail the whole operation if database save fails
        
        logger.info("Channel importance computation with visualizations complete")
        return result
