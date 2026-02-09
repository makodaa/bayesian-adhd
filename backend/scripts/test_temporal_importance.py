"""
Test script for temporal importance analysis.

This script tests the temporal importance service by:
1. Loading sample EEG data
2. Computing temporal importance
3. Displaying results and visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import ELECTRODE_CHANNELS, SAMPLE_RATE
from app.ml.model_loader import ModelLoader
from app.services.temporal_importance_service import TemporalImportanceService
from app.core.logging_config import get_app_logger

logger = get_app_logger(__name__)


def create_sample_eeg_data(duration_seconds: int = 10) -> pd.DataFrame:
    """
    Create sample EEG data for testing.

    Args:
        duration_seconds: Duration of signal in seconds

    Returns:
        DataFrame with 19 electrode channels
    """
    num_samples = duration_seconds * SAMPLE_RATE

    # Create synthetic EEG-like data
    time = np.linspace(0, duration_seconds, num_samples)

    data = {}
    for i, channel in enumerate(ELECTRODE_CHANNELS):
        # Mix of different frequencies (theta, alpha, beta)
        theta = 0.5 * np.sin(2 * np.pi * 6 * time)  # 6 Hz theta
        alpha = 0.3 * np.sin(2 * np.pi * 10 * time)  # 10 Hz alpha
        beta = 0.2 * np.sin(2 * np.pi * 20 * time)   # 20 Hz beta
        
        # Add some noise
        noise = 0.1 * np.random.randn(num_samples)
        
        # Combine with channel-specific phase shift
        phase_shift = i * 0.1
        signal = theta + alpha + beta + noise
        signal = np.roll(signal, int(phase_shift * SAMPLE_RATE))
        
        data[channel] = signal
    
    return pd.DataFrame(data)


def test_temporal_importance():
    """Test temporal importance computation."""
    
    print("=" * 80)
    print("TEMPORAL IMPORTANCE TEST")
    print("=" * 80)
    
    # Initialize model and service
    print("\n1. Initializing model...")
    model_loader = ModelLoader()
    model_loader.initialize()
    print("   ✓ Model loaded")
    
    temporal_service = TemporalImportanceService(model_loader)
    print("   ✓ Temporal importance service initialized")
    
    # Create sample data
    print("\n2. Creating sample EEG data...")
    duration_seconds = 10
    df = create_sample_eeg_data(duration_seconds)
    print(f"   ✓ Created {len(df)} samples ({duration_seconds} seconds)")
    print(f"   ✓ Channels: {list(df.columns)}")
    
    # Test with different parameters
    test_cases = [
        {"window_size_ms": 500, "stride_ms": 100, "name": "Default"},
        {"window_size_ms": 1000, "stride_ms": 200, "name": "Large window"},
        {"window_size_ms": 250, "stride_ms": 50, "name": "Fine-grained"},
    ]
    
    for i, params in enumerate(test_cases, 1):
        print(f"\n3.{i}. Testing: {params['name']}")
        print(f"     Window size: {params['window_size_ms']}ms")
        print(f"     Stride: {params['stride_ms']}ms")
        
        try:
            # Compute temporal importance
            result = temporal_service.compute_temporal_importance_with_classification(
                df,
                window_size_ms=params['window_size_ms'],
                stride_ms=params['stride_ms']
            )
            
            # Display results
            print(f"\n     Classification Results:")
            print(f"     - Predicted class: {result['classification']}")
            print(f"     - Confidence: {result['confidence']:.2%}")
            
            temporal_data = result['temporal_importance']
            print(f"\n     Temporal Analysis:")
            print(f"     - Duration: {temporal_data['total_duration_sec']:.2f} seconds")
            print(f"     - Windows analyzed: {temporal_data['num_windows']}")
            print(f"     - Baseline probability: {temporal_data['baseline_probability']:.4f}")
            
            importance_scores = temporal_data['importance_scores']
            if importance_scores:
                print(f"\n     Importance Statistics:")
                print(f"     - Mean: {np.mean(importance_scores):.4f}")
                print(f"     - Std Dev: {np.std(importance_scores):.4f}")
                print(f"     - Max: {np.max(importance_scores):.4f}")
                print(f"     - Min: {np.min(importance_scores):.4f}")
                
                # Find top 3 important time points
                sorted_indices = np.argsort(importance_scores)[::-1][:3]
                time_points = temporal_data['time_points']
                print(f"\n     Top 3 Important Time Points:")
                for idx in sorted_indices:
                    print(f"     - {time_points[idx]:.2f}s: {importance_scores[idx]:.4f}")
            
            print(f"\n     ✓ {params['name']} test completed successfully")
            
        except Exception as e:
            print(f"\n     ✗ Error: {e}")
            logger.error(f"Test failed for {params['name']}: {e}", exc_info=True)
    
    # Test with visualizations
    print(f"\n4. Testing with visualizations...")
    try:
        result = temporal_service.compute_temporal_importance_with_visualizations(
            df,
            window_size_ms=500,
            stride_ms=100
        )
        
        print(f"   ✓ Visualizations generated:")
        for viz_name in result['visualizations'].keys():
            viz_data = result['visualizations'][viz_name]
            print(f"     - {viz_name}: {len(viz_data)} bytes (base64)")
        
        print(f"\n   ✓ All visualizations created successfully")
        
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        logger.error(f"Visualization test failed: {e}", exc_info=True)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_temporal_importance()
