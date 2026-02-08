# Channel Importance API Documentation

## Overview

The Channel Importance API displays spatial features showing which EEG channels (brain regions) contribute most to the ADHD classification decision. It uses **occlusion sensitivity analysis** to measure the importance of each electrode channel.

## Method: Occlusion Sensitivity

For each EEG channel:

1. Run baseline inference with all channels intact
2. Temporarily zero out the channel (occlude it)
3. Run inference again
4. Calculate importance = baseline_probability - occluded_probability

Higher importance scores indicate channels that are more critical for the classification decision.

## API Endpoints

### 1. Compute Channel Importance from Upload

**Endpoint:** `POST /api/channel-importance/upload`

**Authentication:** Required (login_required)

**Description:** Analyzes an uploaded EEG CSV file and returns channel importance scores.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body:
  - `file`: CSV file containing EEG data with 19 electrode channels

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
  "channel_importance": {
    "Fp1": 0.0234,
    "Fp2": 0.0189,
    "F3": 0.0456,
    "F4": 0.0523,
    "C3": 0.0312,
    "C4": 0.0298,
    "P3": 0.0245,
    "P4": 0.0267,
    "O1": 0.0178,
    "O2": 0.0145,
    "F7": 0.0389,
    "F8": 0.0412,
    "T7": 0.0234,
    "T8": 0.0256,
    "P7": 0.0198,
    "P8": 0.0211,
    "Fz": 0.0445,
    "Cz": 0.0334,
    "Pz": 0.0223
  },
  "importance_normalized": {
    "Fp1": 0.3456,
    "Fp2": 0.2789,
    "F3": 0.6123,
    "F4": 0.7234,
    "C3": 0.4567,
    "C4": 0.4234,
    "P3": 0.3678,
    "P4": 0.3945,
    "O1": 0.2456,
    "O2": 0.1823,
    "F7": 0.5234,
    "F8": 0.5678,
    "T7": 0.3456,
    "T8": 0.3789,
    "P7": 0.2912,
    "P8": 0.3123,
    "Fz": 0.6012,
    "Cz": 0.4789,
    "Pz": 0.3234
  },
  "electrode_positions": {
    "Fp1": [-0.31, 0.95],
    "Fp2": [0.31, 0.95],
    "F7": [-0.75, 0.55],
    "F3": [-0.45, 0.6],
    "Fz": [0.0, 0.65],
    "F4": [0.45, 0.6],
    "F8": [0.75, 0.55],
    "T7": [-0.95, 0.0],
    "C3": [-0.55, 0.0],
    "Cz": [0.0, 0.0],
    "C4": [0.55, 0.0],
    "T8": [0.95, 0.0],
    "P7": [-0.75, -0.55],
    "P3": [-0.45, -0.6],
    "Pz": [0.0, -0.65],
    "P4": [0.45, -0.6],
    "P8": [0.75, -0.55],
    "O1": [-0.31, -0.95],
    "O2": [0.31, -0.95]
  }
}
```

**Status Codes:**

- 200: Success
- 400: Invalid file or validation error
- 401: Unauthorized (not logged in)
- 500: Server error

### 2. Compute Channel Importance from Recording

**Endpoint:** `GET /api/channel-importance/recording/<recording_id>`

**Authentication:** Required (login_required)

**Description:** Analyzes an existing recording from the database and returns channel importance scores.

**Request:**

- Method: GET
- URL Parameter: `recording_id` (integer) - The ID of the recording to analyze

**Response:**
Same as upload endpoint, plus:

```json
{
  "recording_id": 123,
  "subject_id": 456,
  "subject_code": "SUBJ001",
  ...
}
```

**Status Codes:**

- 200: Success
- 400: Validation error
- 401: Unauthorized (not logged in)
- 404: Recording not found or file not found
- 500: Server error

## Response Fields Explained

### Classification Results

- `classification`: Predicted ADHD subtype or Non-ADHD
- `confidence`: Confidence score (0-1) for the predicted class
- `predicted_class_idx`: Index of predicted class (0-3)
- `class_probabilities`: Probability distribution across all 4 classes

### Channel Importance Scores

- `channel_importance`: Raw importance scores (probability drop when channel is removed)
  - Higher values = more important channel
  - Positive values indicate the channel contributes to the predicted class
  - Negative values (rare) indicate the channel may confuse the model
- `importance_normalized`: Normalized importance scores (0-1 range)
  - Scaled for easier visualization
  - 0 = least important channel
  - 1 = most important channel

### Electrode Positions

- `electrode_positions`: Standard 10-20 system coordinates (x, y)
  - x: left (-1) to right (+1)
  - y: back (-1) to front (+1)
  - Use these to create topographic scalp maps

## Visualization Guidelines

### 1. Topographic Heatmap

Use the normalized importance scores with electrode positions to create a scalp map:

```python
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np

# Extract data
positions = [(pos[0], pos[1]) for pos in electrode_positions.values()]
values = list(importance_normalized.values())
labels = list(importance_normalized.keys())

# Create grid
xi = np.linspace(-1, 1, 100)
yi = np.linspace(-1, 1, 100)
zi = griddata(positions, values, (xi[None,:], yi[:,None]), method='cubic')

# Plot
plt.figure(figsize=(8, 8))
plt.contourf(xi, yi, zi, levels=15, cmap='hot')
plt.colorbar(label='Importance Score')

# Add electrode markers
for (x, y), label in zip(positions, labels):
    plt.plot(x, y, 'ko', markersize=8)
    plt.text(x, y, label, ha='center', va='center', fontsize=8)

# Draw head outline
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
plt.axis('equal')
plt.axis('off')
plt.title('Channel Importance Map')
plt.show()
```

### 2. Bar Chart

Display the top N most important channels:

```python
import matplotlib.pyplot as plt

# Sort by importance
sorted_channels = sorted(
    channel_importance.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

channels, scores = zip(*sorted_channels)

plt.figure(figsize=(10, 6))
plt.barh(channels, scores)
plt.xlabel('Importance Score')
plt.ylabel('Channel')
plt.title('Top 10 Most Important EEG Channels')
plt.tight_layout()
plt.show()
```

### 3. Regional Analysis

Group channels by brain region:

```python
regions = {
    'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
    'Central': ['C3', 'C4', 'Cz'],
    'Temporal': ['T7', 'T8'],
    'Parietal': ['P3', 'P4', 'P7', 'P8', 'Pz'],
    'Occipital': ['O1', 'O2']
}

regional_importance = {
    region: np.mean([channel_importance[ch] for ch in channels])
    for region, channels in regions.items()
}

plt.figure(figsize=(8, 6))
plt.bar(regional_importance.keys(), regional_importance.values())
plt.xlabel('Brain Region')
plt.ylabel('Average Importance')
plt.title('Regional Contribution to Classification')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Example Usage

### Using cURL

```bash
# Upload file
curl -X POST http://localhost:5000/api/channel-importance/upload \
  -H "Cookie: session=<session_id>" \
  -F "file=@path/to/eeg_data.csv"

# Analyze existing recording
curl -X GET http://localhost:5000/api/channel-importance/recording/123 \
  -H "Cookie: session=<session_id>"
```

### Using Python requests

```python
import requests

# Login first
session = requests.Session()
session.post('http://localhost:5000/api/auth/login', json={
    'username': 'your_username',
    'password': 'your_password'
})

# Upload file
with open('eeg_data.csv', 'rb') as f:
    response = session.post(
        'http://localhost:5000/api/channel-importance/upload',
        files={'file': f}
    )
    result = response.json()

print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\nTop 5 Important Channels:")
sorted_channels = sorted(
    result['channel_importance'].items(),
    key=lambda x: x[1],
    reverse=True
)[:5]
for channel, importance in sorted_channels:
    print(f"  {channel}: {importance:.4f}")

# Analyze existing recording
response = session.get(
    'http://localhost:5000/api/channel-importance/recording/123'
)
result = response.json()
```

### Using JavaScript (fetch)

```javascript
// Upload file
const formData = new FormData();
formData.append("file", fileInput.files[0]);

fetch("/api/channel-importance/upload", {
  method: "POST",
  body: formData,
  credentials: "include",
})
  .then((response) => response.json())
  .then((data) => {
    console.log("Classification:", data.classification);
    console.log("Confidence:", data.confidence);
    console.log("Channel Importance:", data.channel_importance);

    // Create visualization using the data
    createTopographicMap(data.electrode_positions, data.importance_normalized);
  })
  .catch((error) => console.error("Error:", error));

// Analyze existing recording
fetch("/api/channel-importance/recording/123", {
  credentials: "include",
})
  .then((response) => response.json())
  .then((data) => {
    console.log("Result:", data);
  });
```

## Performance Considerations

- **Computation Time**: The analysis involves running inference 20 times (baseline + 19 occluded versions), so it takes approximately 20x longer than a standard classification
- **Typical Processing Time**: 5-15 seconds depending on recording length and hardware
- **Memory Usage**: Moderate - processes windows sequentially to avoid memory issues
- **Recommendations**:
  - Use for detailed analysis, not real-time classification
  - Consider caching results for frequently analyzed recordings
  - Run asynchronously in production to avoid blocking the UI

## Clinical Interpretation

### High Importance Regions

- **Frontal Channels (Fp1, Fp2, F3, F4, Fz)**: Executive function, attention control
- **Central Channels (C3, C4, Cz)**: Motor control, sensorimotor processing
- **Temporal Channels (T7, T8)**: Auditory processing, memory
- **Parietal Channels (P3, P4, Pz)**: Spatial processing, attention
- **Occipital Channels (O1, O2)**: Visual processing

### Expected Patterns for ADHD

Research suggests ADHD is associated with:

- Increased theta activity in frontal regions
- Decreased beta activity in frontal and central regions
- Elevated theta/beta ratio (TBR)

The channel importance scores may highlight:

- Frontal and central regions for ADHD-I (Inattentive)
- Motor and central regions for ADHD-H (Hyperactive-Impulsive)
- Distributed pattern for ADHD-C (Combined)

## Technical Details

### Algorithm: Occlusion Sensitivity

```
For each channel i in [1..19]:
    1. baseline_prob = P(class | all_channels)
    2. occluded_data = zero_out(channel_i, data)
    3. occluded_prob = P(class | occluded_data)
    4. importance_i = baseline_prob - occluded_prob

Normalization:
    normalized_i = (importance_i - min_importance) / (max_importance - min_importance)
```

### Advantages of Occlusion Sensitivity

1. **Model-Agnostic**: Works with any trained model
2. **Intuitive**: Direct measure of feature contribution
3. **No Gradient Required**: Unlike gradient-based methods
4. **Robust**: Not affected by gradient saturation

### Limitations

1. **Computational Cost**: Requires multiple forward passes
2. **Independence Assumption**: Doesn't capture channel interactions
3. **Occlusion Effect**: Zeroing may create unnatural patterns

## Troubleshooting

### Error: "Missing required channels"

- Ensure CSV has all 19 standard EEG channels
- Check column names match exactly: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz

### Error: "Recording file not found"

- Verify the recording exists in the database
- Check that the file_path in the database is correct
- Ensure the file hasn't been moved or deleted

### Low Importance Scores

- Normal if the model is uncertain about the classification
- Check the confidence score - low confidence may indicate ambiguous data
- Review data quality - artifacts can affect importance scores

### Unexpected Importance Pattern

- Consider the specific ADHD subtype - patterns vary
- Review raw classification probabilities
- Compare with clinical assessment

## Future Enhancements

Potential additions to the API:

1. **Integrated Gradients**: More accurate gradient-based attribution
2. **Layer-wise Relevance Propagation (LRP)**: Backpropagate relevance scores
3. **SHAP Values**: Game-theoretic feature importance
4. **Temporal Importance**: Show how importance changes over time
5. **Interactive Visualization**: Built-in 3D brain visualization
6. **Batch Processing**: Analyze multiple recordings at once
7. **Statistical Significance**: Test if importance scores are significant
