# Temporal Importance API Documentation

## Overview

The Temporal Importance API displays **temporal features** showing when (in time) and at what rhythms the model is most sensitive. It uses **sliding window occlusion sensitivity analysis** to measure the importance of different time segments in the EEG signal for ADHD classification.

## Method: Temporal Occlusion Sensitivity

For each time window in the EEG signal:

1. Run baseline inference with all data intact
2. Slide a temporal occlusion window (e.g., 500ms) across the signal
3. Mask each window position (set values to zero)
4. Run inference again
5. Calculate importance = baseline_probability - occluded_probability

Higher importance scores indicate time segments that are more critical for the classification decision.

## Clinical Relevance

### What This Demonstrates

- **Model focuses on specific temporal segments**, not the entire signal uniformly
- Identifies **critical time windows** where diagnostic patterns appear
- Reveals **temporal dynamics** of ADHD-related EEG patterns

### ADHD Context

ADHD is characterized by temporal variations in:

- **Sustained attention**: Periods of focus vs. distraction
- **Behavioral inhibition**: Time-varying self-regulation
- **Neural oscillations**: Dynamic changes in brain rhythms

This analysis bridges deep learning and classical EEG analysis by showing **when** the model detects patterns, not just **what** patterns it detects.

## API Endpoints

### 1. Compute Temporal Importance from Upload

**Endpoint:** `POST /api/temporal-importance/upload`

**Authentication:** Required (login_required)

**Description:** Analyzes an uploaded EEG CSV file and returns temporal importance scores.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body:
  - `file`: CSV file containing EEG data with 19 electrode channels
  - `window_size_ms` (optional): Size of occlusion window in milliseconds (default: 500)
  - `stride_ms` (optional): Stride for sliding window in milliseconds (default: 100)

**Response:**

```json
{
  "classification": "Combined / C (ADHD-C)",
  "confidence": 0.7234,
  "predicted_class_idx": 0,
  "class_probabilities": {
    "Combined / C (ADHD-C)": 0.7234,
    "Hyperactive-Impulsive (ADHD-H)": 0.1234,
    "Inattentive (ADHD-I)": 0.0987,
    "Non-ADHD": 0.0545
  },
  "temporal_importance": {
    "time_points": [0.0, 0.1, 0.2, 0.3, ...],
    "importance_scores": [0.0234, 0.0189, 0.0456, 0.0312, ...],
    "baseline_probability": 0.7234,
    "predicted_class_idx": 0,
    "window_size_ms": 500,
    "stride_ms": 100,
    "total_duration_sec": 60.0,
    "num_windows": 595
  },
  "visualizations": {
    "time_curve": "data:image/png;base64,...",
    "heatmap": "data:image/png;base64,...",
    "statistics": "data:image/png;base64,..."
  }
}
```

**Example cURL:**

```bash
curl -X POST \
  -H "Cookie: session=..." \
  -F "file=@eeg_data.csv" \
  -F "window_size_ms=500" \
  -F "stride_ms=100" \
  http://localhost:5000/api/temporal-importance/upload
```

**Example Python:**

```python
import requests

with open('eeg_data.csv', 'rb') as f:
    files = {'file': f}
    data = {
        'window_size_ms': 500,
        'stride_ms': 100
    }
    response = requests.post(
        'http://localhost:5000/api/temporal-importance/upload',
        files=files,
        data=data,
        cookies={'session': 'your_session_cookie'}
    )
    result = response.json()

    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.2%}")

    # Access temporal data
    temporal = result['temporal_importance']
    print(f"Analyzed {temporal['num_windows']} time windows")
    print(f"Duration: {temporal['total_duration_sec']} seconds")
```

---

### 2. Compute Temporal Importance from Recording

**Endpoint:** `GET /api/temporal-importance/recording/<recording_id>`

**Authentication:** Required (login_required)

**Description:** Computes temporal importance for an existing recording in the database. Results are automatically saved to the database.

**Parameters:**

- `recording_id` (path): ID of the recording
- `window_size_ms` (query, optional): Size of occlusion window in milliseconds (default: 500)
- `stride_ms` (query, optional): Stride for sliding window in milliseconds (default: 100)

**Response:**

Same as upload endpoint, with additional fields:

```json
{
  "classification": "Combined / C (ADHD-C)",
  "confidence": 0.7234,
  "predicted_class_idx": 0,
  "class_probabilities": {...},
  "temporal_importance": {...},
  "visualizations": {...},
  "recording_id": 42,
  "subject_id": 7,
  "subject_code": "SUB-001",
  "result_id": 123
}
```

**Example cURL:**

```bash
curl -X GET \
  -H "Cookie: session=..." \
  "http://localhost:5000/api/temporal-importance/recording/42?window_size_ms=500&stride_ms=100"
```

**Example Python:**

```python
import requests

response = requests.get(
    'http://localhost:5000/api/temporal-importance/recording/42',
    params={
        'window_size_ms': 500,
        'stride_ms': 100
    },
    cookies={'session': 'your_session_cookie'}
)
result = response.json()

# Extract temporal importance data
temporal = result['temporal_importance']
time_points = temporal['time_points']
importance_scores = temporal['importance_scores']

# Plot with matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(time_points, importance_scores)
plt.xlabel('Time (seconds)')
plt.ylabel('Importance Score')
plt.title(f"Temporal Importance - {result['classification']}")
plt.grid(True, alpha=0.3)
plt.show()
```

---

### 3. Retrieve Temporal Importance from Database

**Endpoint:** `GET /api/temporal-importance/result/<result_id>`

**Authentication:** Required (login_required)

**Description:** Retrieves previously computed temporal importance data from the database for a specific result.

**Parameters:**

- `result_id` (path): ID of the result

**Response:**

```json
{
  "result_id": 123,
  "classification": "Combined / C (ADHD-C)",
  "confidence": 0.7234,
  "temporal_importance": {
    "time_points": [0.0, 0.1, 0.2, ...],
    "importance_scores": [0.0234, 0.0189, ...],
    "window_size_ms": 500,
    "stride_ms": 100,
    "num_windows": 595,
    "total_duration_sec": 60.0
  },
  "visualizations": {
    "time_curve": "data:image/png;base64,...",
    "heatmap": "data:image/png;base64,...",
    "statistics": "data:image/png;base64,..."
  },
  "created_at": "2026-02-09T10:30:00"
}
```

**Example cURL:**

```bash
curl -X GET \
  -H "Cookie: session=..." \
  http://localhost:5000/api/temporal-importance/result/123
```

**Example Python:**

```python
import requests
import base64
from io import BytesIO
from PIL import Image

response = requests.get(
    'http://localhost:5000/api/temporal-importance/result/123',
    cookies={'session': 'your_session_cookie'}
)
result = response.json()

# Display visualizations
for viz_name, viz_data in result['visualizations'].items():
    # Remove data:image/png;base64, prefix
    img_data = viz_data.split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes))
    img.show()
    print(f"Displayed: {viz_name}")
```

---

## Response Fields

### Classification Fields

| Field                 | Type    | Description                                                |
| --------------------- | ------- | ---------------------------------------------------------- |
| `classification`      | string  | Predicted ADHD subtype name                                |
| `confidence`          | float   | Confidence score (0-1) for prediction                      |
| `predicted_class_idx` | integer | Class index (0: ADHD-C, 1: ADHD-H, 2: ADHD-I, 3: Non-ADHD) |
| `class_probabilities` | object  | Probability for each class                                 |

### Temporal Importance Fields

| Field                  | Type         | Description                              |
| ---------------------- | ------------ | ---------------------------------------- |
| `time_points`          | array[float] | Time points in seconds                   |
| `importance_scores`    | array[float] | Importance score at each time point      |
| `baseline_probability` | float        | Baseline prediction probability          |
| `predicted_class_idx`  | integer      | Target class for importance calculation  |
| `window_size_ms`       | integer      | Size of occlusion window (milliseconds)  |
| `stride_ms`            | integer      | Stride for sliding window (milliseconds) |
| `total_duration_sec`   | float        | Total duration of signal (seconds)       |
| `num_windows`          | integer      | Number of temporal windows analyzed      |

### Visualization Fields

| Field        | Type   | Description                                              |
| ------------ | ------ | -------------------------------------------------------- |
| `time_curve` | string | Base64 PNG: Line plot of importance over time            |
| `heatmap`    | string | Base64 PNG: Heatmap visualization of temporal importance |
| `statistics` | string | Base64 PNG: Statistical summary plots                    |

---

## Visualizations

### 1. Time-Importance Curve

**Description:** Line plot showing importance scores over time.

**Features:**

- **X-axis:** Time (seconds)
- **Y-axis:** Importance score (probability drop)
- **Highlighted regions:** Time windows with above-average importance
- **Zero line:** Reference for neutral importance

**Interpretation:**

- **Positive peaks:** Time segments critical for classification
- **Near-zero regions:** Less important for decision
- **Negative values:** Segments that reduce confidence (rare)

### 2. Temporal Heatmap

**Description:** Color-coded heatmap of importance across time.

**Features:**

- **Color scale:** Red (high importance) to Blue (low importance)
- **X-axis:** Time progression
- **Single row:** Condensed view for quick pattern identification

**Interpretation:**

- **Hot spots (red):** Critical temporal segments
- **Cool regions (blue):** Less relevant time periods

### 3. Statistical Summary

**Description:** Two-panel plot with distribution and summary statistics.

**Left Panel - Distribution:**

- Histogram of importance scores
- Mean line (red dashed)
- Median line (green dashed)

**Right Panel - Statistics:**

- Total duration
- Baseline probability
- Mean importance
- Standard deviation
- Max/min importance
- Top 3 most important time points

---

## Parameters Guide

### Window Size (`window_size_ms`)

**Default:** 500ms (0.5 seconds)

**Purpose:** Size of the temporal window to occlude.

**Recommendations:**

- **Small windows (100-300ms):** Fine-grained temporal resolution, detects brief patterns
- **Medium windows (500-1000ms):** Balanced resolution and computational efficiency
- **Large windows (1000-2000ms):** Coarse resolution, identifies broad temporal patterns

**Trade-offs:**

- Smaller windows → More detail, longer computation
- Larger windows → Less detail, faster computation

### Stride (`stride_ms`)

**Default:** 100ms (0.1 seconds)

**Purpose:** Step size for sliding the occlusion window.

**Recommendations:**

- **Small stride (50-100ms):** Smooth importance curve, overlapping analysis
- **Medium stride (100-200ms):** Good balance
- **Large stride (300-500ms):** Faster computation, less smooth curve

**Trade-offs:**

- Smaller stride → Smoother curve, more windows, longer computation
- Larger stride → Coarser sampling, fewer windows, faster computation

**Relationship to Window Size:**

- Typical ratio: stride = 20-30% of window size
- Example: 500ms window with 100ms stride = 80% overlap

---

## Use Cases

### 1. Clinical Assessment

**Scenario:** Clinician reviews temporal patterns for diagnosis.

**Workflow:**

1. Upload or select patient EEG recording
2. Compute temporal importance with default parameters
3. Review time-importance curve for critical segments
4. Identify time windows with high diagnostic value
5. Correlate with behavioral observations during recording

**Value:** Provides temporal context for ADHD symptoms.

### 2. Research Analysis

**Scenario:** Researcher studies temporal dynamics of ADHD patterns.

**Workflow:**

1. Compute temporal importance for dataset of recordings
2. Compare temporal patterns across ADHD subtypes
3. Analyze when in time diagnostic patterns emerge
4. Study relationship to attention tasks or stimuli

**Value:** Reveals temporal structure of ADHD-related brain activity.

### 3. Model Validation

**Scenario:** Validate that model focuses on clinically relevant time periods.

**Workflow:**

1. Compute temporal importance for known cases
2. Verify model focuses on expected temporal patterns
3. Check for artifacts or spurious temporal dependencies
4. Ensure temporal consistency across similar recordings

**Value:** Confirms model interpretability and clinical validity.

### 4. Data Quality Assessment

**Scenario:** Identify problematic temporal segments in recordings.

**Workflow:**

1. Compute temporal importance
2. Check for unexpected importance patterns
3. Identify potential artifacts (spikes in importance)
4. Validate data quality before diagnosis

**Value:** Helps detect recording issues or artifacts.

---

## Implementation Details

### Algorithm

```
1. Load EEG data (19 channels × N samples)
2. Compute baseline prediction (full signal)
3. For each time window position:
   a. Create copy of EEG data
   b. Mask window (set to zero)
   c. Compute prediction with masked data
   d. Calculate importance = baseline_prob - masked_prob
4. Return time series of importance scores
```

### Computational Complexity

- **Time Complexity:** O(W × C) where W = number of windows, C = cost of inference
- **Space Complexity:** O(N × Ch) where N = samples, Ch = 19 channels
- **Typical Runtime:** 10-30 seconds for 60-second recording (depends on parameters)

### Preprocessing

Same preprocessing as main classification pipeline:

1. Bandpass filter (0.5-60 Hz)
2. ICA artifact removal
3. Normalization (z-score)
4. Windowing and frequency analysis

---

## Error Handling

### Common Errors

**400 Bad Request - Missing File**

```json
{ "error": "No file provided" }
```

**400 Bad Request - Invalid File Type**

```json
{ "error": "File type not supported" }
```

**400 Bad Request - Validation Error**

```json
{ "error": "Missing required channels: ['Fp1', 'Fp2']" }
```

**404 Not Found - Recording**

```json
{ "error": "Recording not found" }
```

**404 Not Found - Temporal Data**

```json
{ "error": "Temporal importance data not found" }
```

**500 Internal Server Error**

```json
{ "error": "Failed to compute temporal importance: <details>" }
```

---

## Best Practices

### 1. Parameter Selection

- Start with default parameters (500ms window, 100ms stride)
- Adjust based on signal duration and computational resources
- Ensure stride ≤ window_size for overlap

### 2. Interpretation

- Focus on relative importance, not absolute values
- Compare temporal patterns across recordings
- Consider baseline probability when interpreting scores
- Look for consistent patterns in similar cases

### 3. Performance

- Longer recordings → longer computation time
- Smaller stride → more windows → longer computation
- Consider caching results in database (use recording endpoint)

### 4. Data Quality

- Ensure clean EEG data (artifacts affect temporal analysis)
- Verify 19 electrode channels present
- Check for consistent sampling rate (128 Hz)

---

## Database Schema

```sql
CREATE TABLE temporal_importance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    time_points TEXT NOT NULL,  -- JSON array
    importance_scores TEXT NOT NULL,  -- JSON array
    window_size_ms INTEGER NOT NULL,
    stride_ms INTEGER NOT NULL,
    time_curve_plot TEXT,  -- Base64 PNG
    heatmap_plot TEXT,  -- Base64 PNG
    statistics_plot TEXT,  -- Base64 PNG
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (result_id) REFERENCES results(id) ON DELETE CASCADE
);
```

---

## Advanced Usage

### Custom Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Get temporal importance data
response = requests.get('http://localhost:5000/api/temporal-importance/result/123')
data = response.json()

time_points = np.array(data['temporal_importance']['time_points'])
importance = np.array(data['temporal_importance']['importance_scores'])

# Find peaks in importance
peaks, properties = find_peaks(importance, prominence=0.01)
peak_times = time_points[peaks]
peak_importances = importance[peaks]

# Plot with annotated peaks
plt.figure(figsize=(14, 6))
plt.plot(time_points, importance, 'b-', linewidth=2, label='Importance')
plt.plot(peak_times, peak_importances, 'ro', markersize=10, label='Peaks')

for pt, pi in zip(peak_times, peak_importances):
    plt.annotate(f'{pt:.1f}s', xy=(pt, pi), xytext=(5, 5),
                textcoords='offset points', fontsize=9)

plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)
plt.title('Temporal Importance with Peak Detection', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Found {len(peaks)} important time segments:")
for i, (pt, pi) in enumerate(zip(peak_times, peak_importances), 1):
    print(f"  {i}. Time: {pt:.2f}s, Importance: {pi:.4f}")
```

### Comparative Analysis

```python
# Compare temporal importance across multiple subjects
subject_ids = [101, 102, 103, 104]
temporal_data = []

for sid in subject_ids:
    # Get recording for subject
    recording_response = requests.get(
        f'http://localhost:5000/api/recordings/subject/{sid}'
    )
    recording_id = recording_response.json()[0]['id']

    # Get temporal importance
    ti_response = requests.get(
        f'http://localhost:5000/api/temporal-importance/recording/{recording_id}'
    )
    temporal_data.append(ti_response.json())

# Plot all subjects
fig, axes = plt.subplots(len(subject_ids), 1, figsize=(14, 3*len(subject_ids)))

for ax, data, sid in zip(axes, temporal_data, subject_ids):
    ti = data['temporal_importance']
    ax.plot(ti['time_points'], ti['importance_scores'])
    ax.set_title(f"Subject {sid} - {data['classification']}")
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Importance')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Related APIs

- **Channel Importance API:** `/api/channel-importance/*` - Spatial analysis (which brain regions)
- **Classification API:** `/api/classify/*` - ADHD subtype classification
- **Results API:** `/api/results/*` - Classification results management

---

## Version History

| Version | Date       | Changes                                                      |
| ------- | ---------- | ------------------------------------------------------------ |
| 1.0.0   | 2026-02-09 | Initial release with temporal occlusion sensitivity analysis |

---

## Support

For questions or issues:

- Check error messages for specific guidance
- Verify EEG data format (19 channels, CSV)
- Ensure valid session authentication
- Review parameter ranges (window_size_ms > 0, stride_ms > 0)

## Citation

If using this API in research, please cite:

```
Temporal Importance Analysis for ADHD Classification
Bayesian ADHD EEG Analysis System, 2026
```
