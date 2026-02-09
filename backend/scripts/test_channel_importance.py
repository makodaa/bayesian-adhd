"""
Test script for Channel Importance Service

This script tests the channel importance calculation on sample data
to ensure the implementation works correctly.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import ELECTRODE_CHANNELS, SAMPLE_RATE
from app.ml.model_loader import ModelLoader
from app.services.channel_importance_service import ChannelImportanceService


def generate_sample_eeg_data(duration_seconds=10, sample_rate=SAMPLE_RATE):
    """
    Generate synthetic EEG data for testing.

    Args:
        duration_seconds: Length of recording in seconds
        sample_rate: Sampling rate in Hz

    Returns:
        DataFrame with 19 EEG channels
    """
    n_samples = duration_seconds * sample_rate
    time = np.linspace(0, duration_seconds, n_samples)

    data = {}
    for channel in ELECTRODE_CHANNELS:
        # Generate synthetic EEG-like signal (combination of sine waves)
        # Add different frequency components (theta, alpha, beta)
        theta = 0.5 * np.sin(2 * np.pi * 5 * time)  # 5 Hz
        alpha = 0.3 * np.sin(2 * np.pi * 10 * time)  # 10 Hz
        beta = 0.2 * np.sin(2 * np.pi * 20 * time)  # 20 Hz
        noise = 0.1 * np.random.randn(n_samples)

        signal = theta + alpha + beta + noise
        data[channel] = signal

    return pd.DataFrame(data)


def test_channel_importance_service():
    """Test the channel importance service with synthetic data."""

    print("=" * 80)
    print("CHANNEL IMPORTANCE SERVICE TEST")
    print("=" * 80)
    print()

    # Initialize model loader
    print("1. Initializing model...")
    try:
        model_loader = ModelLoader()
        model_loader.initialize()
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False
    
    # Initialize channel importance service
    print("\n2. Initializing channel importance service...")
    try:
        service = ChannelImportanceService(model_loader)
        print("   ✓ Service initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize service: {e}")
        return False
    
    # Generate sample data
    print("\n3. Generating sample EEG data...")
    try:
        df = generate_sample_eeg_data(duration_seconds=10)
        print(f"   ✓ Generated {len(df)} samples across {len(df.columns)} channels")
        print(f"   Channels: {df.columns.tolist()}")
    except Exception as e:
        print(f"   ✗ Failed to generate data: {e}")
        return False
    
    # Test basic channel importance computation
    print("\n4. Computing channel importance (basic)...")
    try:
        importance_scores = service.compute_channel_importance(df)
        print(f"   ✓ Computed importance for {len(importance_scores)} channels")
        
        # Show top 5 channels
        sorted_channels = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        print("\n   Top 5 Most Important Channels:")
        for channel, importance in sorted_channels:
            print(f"   - {channel:4s}: {importance:.6f}")
            
    except Exception as e:
        print(f"   ✗ Failed to compute importance: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test full analysis with classification
    print("\n5. Computing channel importance with classification and visualizations...")
    try:
        result = service.compute_channel_importance_with_visualizations(df)
        
        print(f"   ✓ Analysis complete")
        print(f"\n   Classification Results:")
        print(f"   - Predicted Class: {result['classification']}")
        print(f"   - Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"   - Class Index: {result['predicted_class_idx']}")
        
        print(f"\n   Class Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"   - {class_name:35s}: {prob:.4f} ({prob*100:.2f}%)")
        
        print(f"\n   Top 5 Important Channels (Normalized):")
        sorted_normalized = sorted(
            result['importance_normalized'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for channel, importance in sorted_normalized:
            print(f"   - {channel:4s}: {importance:.6f}")
        
        # Check visualizations
        print(f"\n   Visualizations:")
        if 'visualizations' in result:
            viz = result['visualizations']
            for viz_name in ['topographic_map', 'bar_chart', 'regional_chart']:
                if viz_name in viz and viz[viz_name].startswith('data:image/png;base64,'):
                    base64_len = len(viz[viz_name])
                    print(f"   - {viz_name.replace('_', ' ').title()}: ✓ ({base64_len} chars)")
                else:
                    print(f"   - {viz_name.replace('_', ' ').title()}: ✗ Missing or invalid")
        else:
            print(f"   ✗ No visualizations found in result")
        
        # Validate response structure
        required_keys = [
            'classification', 'confidence', 'predicted_class_idx',
            'class_probabilities', 'channel_importance', 'importance_normalized',
            'visualizations'
        ]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            print(f"\n   ⚠ Warning: Missing keys in result: {missing_keys}")
        else:
            print(f"\n   ✓ All required keys present in result")
            
    except Exception as e:
        print(f"   ✗ Failed to compute full analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate importance scores
    print("\n6. Validating importance scores...")
    try:
        # Check that all channels have importance scores
        if len(result['channel_importance']) != len(ELECTRODE_CHANNELS):
            print(f"   ✗ Expected {len(ELECTRODE_CHANNELS)} channels, got {len(result['channel_importance'])}")
            return False
        
        # Check that normalized scores are in [0, 1] range
        normalized_values = list(result['importance_normalized'].values())
        if not all(0 <= v <= 1 for v in normalized_values):
            print(f"   ✗ Normalized scores not in [0, 1] range")
            return False
        
        # Check that at least one score is 0 and one is 1 (after normalization)
        min_normalized = min(normalized_values)
        max_normalized = max(normalized_values)
        if not (abs(min_normalized - 0.0) < 0.01 and abs(max_normalized - 1.0) < 0.01):
            print(f"   ⚠ Warning: Normalization may be off (min={min_normalized:.4f}, max={max_normalized:.4f})")
        
        print(f"   ✓ Importance scores validated")
        print(f"   - Raw score range: [{min(result['channel_importance'].values()):.6f}, {max(result['channel_importance'].values()):.6f}]")
        print(f"   - Normalized range: [{min_normalized:.6f}, {max_normalized:.6f}]")
        
    except Exception as e:
        print(f"   ✗ Validation failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_channel_importance_service()
    sys.exit(0 if success else 1)
