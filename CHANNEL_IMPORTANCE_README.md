# Channel Importance Feature - Quick Start

## What It Does

Shows which EEG channels (brain regions) contribute most to the ADHD classification decision using **occlusion sensitivity analysis**. Generates **MNE-based topographic heatmaps** and charts as base64-encoded images.

## Quick Start

### 1. Start the Server

```bash
cd backend
python app/main.py
```

### 2. Access the Demo

Navigate to: `http://localhost:5000/channel-importance-demo.html`

### 3. Upload EEG Data

- Click "Choose File"
- Select a CSV file with 19 EEG channels
- Click "Analyze Channel Importance"
- Wait 10-15 seconds for results

### 4. View Results

Four visualization tabs:

- **Topographic Map**: Brain scalp heatmap
- **Bar Chart**: Channel rankings
- **Regional Analysis**: Grouped by brain region
- **Data Table**: Detailed scores

## API Endpoints

### Analyze Uploaded File

```bash
curl -X POST http://localhost:5000/api/channel-importance/upload \
  -H "Cookie: session=<session_id>" \
  -F "file=@eeg_data.csv"
```

### Analyze Existing Recording

```bash
curl -X GET http://localhost:5000/api/channel-importance/recording/123 \
  -H "Cookie: session=<session_id>"
```

## Response Format

```json
{
  "classification": "Combined / C (ADHD-C)",
  "confidence": 0.7234,
  "channel_importance": {
    "Fp1": 0.0234,
    "F4": 0.0523,
    ...
  },
  "importance_normalized": {
    "Fp1": 0.3456,
    "F4": 0.7234,
    ...
  },
  "electrode_positions": {
    "Fp1": [-0.31, 0.95],
    ...
  },
  "visualizations": {
    "topographic_map": "data:image/png;base64,iVBORw0KGgo...",
    "bar_chart": "data:image/png;base64,iVBORw0KGgo...",
    "regional_chart": "data:image/png;base64,iVBORw0KGgo..."
  }
}
```

## Visualizations

All visualizations are generated on the backend using **MNE-Python** and **matplotlib**, then returned as base64-encoded PNG images:

- **Topographic Map**: MNE topomap showing channel importance as a scalp heatmap
- **Bar Chart**: Sorted horizontal bar chart of all 19 channels
- **Regional Chart**: Average importance by brain region (Frontal, Central, Temporal, Parietal, Occipital)

## Test the Implementation

```bash
cd backend
python scripts/test_channel_importance.py
```

## Documentation

- Full API docs: [`backend/CHANNEL_IMPORTANCE_API.md`](backend/CHANNEL_IMPORTANCE_API.md)
- Implementation details: [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md)

## How It Works

For each of the 19 EEG channels:

1. Run baseline prediction with all channels
2. Zero out the channel (occlude it)
3. Run prediction again
4. Calculate: importance = baseline_probability - occluded_probability

Higher scores = more important channels for the classification.

## Brain Regions

- **Frontal** (Fp1, Fp2, F3, F4, F7, F8, Fz): Attention, executive function
- **Central** (C3, C4, Cz): Motor control
- **Temporal** (T7, T8): Auditory, memory
- **Parietal** (P3, P4, P7, P8, Pz): Spatial processing
- **Occipital** (O1, O2): Visual processing

## Clinical Use

The channel importance map helps clinicians:

- Understand which brain regions drive the classification
- Identify atypical activity patterns
- Explain results to patients
- Guide treatment planning

## Performance

- **Processing Time**: 10-15 seconds per recording
- **Why?** Runs inference 20 times (once per channel + baseline)
- **Recommendation**: Use for detailed analysis, not real-time classification

## Requirements

- Logged-in user (authentication required)
- CSV file with 19 standard EEG channels
- Valid EEG data format

## Troubleshooting

**Error: "Missing required channels"**

- Ensure CSV has all 19 channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz

**Analysis takes too long**

- Normal! Processing time is 10-15 seconds
- The algorithm runs 20 model inferences

**Unexpected importance pattern**

- Different ADHD subtypes show different patterns
- Check the confidence score - low confidence may indicate ambiguous data

## Example Code

### Python

```python
import requests

session = requests.Session()
session.post('http://localhost:5000/api/auth/login',
             json={'username': 'user', 'password': 'pass'})

with open('eeg.csv', 'rb') as f:
    response = session.post(
        'http://localhost:5000/api/channel-importance/upload',
        files={'file': f}
    )
    result = response.json()

print(f"Classification: {result['classification']}")
print(f"Top channel: {max(result['channel_importance'].items(), key=lambda x: x[1])}")
```

### JavaScript

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

fetch("/api/channel-importance/upload", {
  method: "POST",
  body: formData,
  credentials: "include",
})
  .then((r) => r.json())
  .then((data) => {
    console.log("Classification:", data.classification);
    console.log("Confidence:", data.confidence);
  });
```

## Next Steps

1. Try the demo page
2. Upload your own EEG data
3. Interpret the topographic map
4. Compare importance across different subjects
5. Use the API in your own applications

For detailed documentation, see [`backend/CHANNEL_IMPORTANCE_API.md`](backend/CHANNEL_IMPORTANCE_API.md).
