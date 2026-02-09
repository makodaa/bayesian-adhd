# Temporal Importance API - Quick Start Guide

## Setup (Run Once)

### 1. Run Database Migration

Connect to your PostgreSQL database and run the migration:

```bash
# Option 1: Using psql
psql -U db_user -d bayesian_adhd -f database/migrations/003_add_temporal_importance.sql

# Option 2: Using Docker (if using docker-compose)
docker-compose exec db psql -U db_user -d bayesian_adhd -f /migrations/003_add_temporal_importance.sql
```

### 2. Verify Installation

Check that the table was created:

```sql
\dt temporal_importance  -- List table
\d temporal_importance   -- Show schema
```

---

## Testing

### Run Test Script

```bash
cd backend
python scripts/test_temporal_importance.py
```

Expected output:

- Model initialization success
- Sample EEG data created
- 3 test cases with different parameters
- Classification results
- Temporal analysis statistics
- Visualization generation success

---

## API Usage

### Start the Server

```bash
cd backend
python -m app.main
```

Server runs on `http://localhost:5000`

### Test Endpoints

#### 1. Upload and Analyze

```bash
curl -X POST http://localhost:5000/api/temporal-importance/upload \
  -H "Cookie: session=YOUR_SESSION_COOKIE" \
  -F "file=@path/to/eeg_data.csv" \
  -F "window_size_ms=500" \
  -F "stride_ms=100"
```

#### 2. Analyze Existing Recording

```bash
curl -X GET "http://localhost:5000/api/temporal-importance/recording/1?window_size_ms=500&stride_ms=100" \
  -H "Cookie: session=YOUR_SESSION_COOKIE"
```

#### 3. Retrieve Saved Analysis

```bash
curl -X GET http://localhost:5000/api/temporal-importance/result/1 \
  -H "Cookie: session=YOUR_SESSION_COOKIE"
```

---

## Python Client Examples

### Example 1: Upload and Analyze

```python
import requests

session_cookie = "YOUR_SESSION_COOKIE"

# Upload CSV file
with open('patient_eeg.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/temporal-importance/upload',
        files={'file': f},
        data={
            'window_size_ms': 500,
            'stride_ms': 100
        },
        cookies={'session': session_cookie}
    )

if response.status_code == 200:
    result = response.json()
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.2%}")

    temporal = result['temporal_importance']
    print(f"Analyzed {temporal['num_windows']} windows")
    print(f"Duration: {temporal['total_duration_sec']:.1f} seconds")
else:
    print(f"Error: {response.json()['error']}")
```

### Example 2: Visualize Results

```python
import matplotlib.pyplot as plt
import numpy as np

# Get temporal importance data
response = requests.get(
    'http://localhost:5000/api/temporal-importance/result/1',
    cookies={'session': session_cookie}
)

data = response.json()
temporal = data['temporal_importance']

# Extract data
time_points = np.array(temporal['time_points'])
importance = np.array(temporal['importance_scores'])

# Plot
plt.figure(figsize=(14, 6))
plt.plot(time_points, importance, linewidth=2)
plt.fill_between(time_points, importance, alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)
plt.title(f"Temporal Importance - {data['classification']}", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('temporal_importance.png', dpi=150)
plt.show()

print(f"Saved plot to temporal_importance.png")
```

### Example 3: Find Peak Importance Times

```python
from scipy.signal import find_peaks

# Get data
response = requests.get(
    'http://localhost:5000/api/temporal-importance/result/1',
    cookies={'session': session_cookie}
)
temporal = response.json()['temporal_importance']

time_points = np.array(temporal['time_points'])
importance = np.array(temporal['importance_scores'])

# Find peaks
peaks, properties = find_peaks(importance, prominence=0.01)

print("Most Important Time Segments:")
for i, peak_idx in enumerate(peaks, 1):
    time = time_points[peak_idx]
    score = importance[peak_idx]
    print(f"{i}. Time: {time:.2f}s, Importance: {score:.4f}")
```

---

## Parameter Recommendations

### Window Size (`window_size_ms`)

| Use Case        | Window Size | Description                            |
| --------------- | ----------- | -------------------------------------- |
| **Fine Detail** | 100-300ms   | Detect brief patterns, high resolution |
| **Balanced** ✓  | 500-1000ms  | Good balance (recommended)             |
| **Overview**    | 1000-2000ms | Broad patterns, faster computation     |

### Stride (`stride_ms`)

| Use Case           | Stride    | Description                          |
| ------------------ | --------- | ------------------------------------ |
| **Smooth Curve** ✓ | 50-100ms  | High overlap, detailed (recommended) |
| **Balanced**       | 100-200ms | Good balance                         |
| **Fast**           | 300-500ms | Less overlap, faster                 |

**Rule of thumb**: stride = 20-30% of window_size

---

## Response Structure

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
    "time_points": [0.0, 0.1, 0.2, ...],
    "importance_scores": [0.023, 0.018, ...],
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

---

## Troubleshooting

### Error: "No file provided"

**Solution**: Ensure you're using `multipart/form-data` and the field name is `file`.

### Error: "Missing required channels"

**Solution**: CSV must have all 19 electrode channels (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz).

### Error: "Recording not found"

**Solution**: Verify the recording ID exists in the database.

### Slow Performance

**Solution**: Increase stride (e.g., 200ms) or reduce window size. Computation time ≈ (signal_duration / stride) × inference_time.

### Connection Refused

**Solution**: Ensure Flask server is running and you're authenticated (valid session cookie).

---

## Files Created

```
backend/
├── app/
│   ├── services/
│   │   └── temporal_importance_service.py    ✅ 750 lines
│   ├── db/
│   │   └── repositories/
│   │       └── temporal_importance.py         ✅ 300 lines
│   └── main.py                                 ✅ Modified
├── scripts/
│   └── test_temporal_importance.py            ✅ 200 lines
├── TEMPORAL_IMPORTANCE_API.md                 ✅ 1000+ lines
├── TEMPORAL_IMPORTANCE_IMPLEMENTATION.md      ✅ 500+ lines
└── database/
    └── migrations/
        └── 003_add_temporal_importance.sql    ✅ 30 lines
```

---

## Next Steps

### 1. Frontend Integration

Create UI components:

- Line chart for temporal importance curve
- Heatmap visualization
- Time segment annotations
- Compare multiple recordings

### 2. Advanced Features

Implement:

- **Band-specific temporal analysis**: Analyze theta, alpha, beta separately
- **Multi-channel temporal analysis**: Per-channel temporal importance
- **Statistical significance testing**: Identify truly important segments
- **Batch processing**: Analyze multiple recordings

### 3. Clinical Workflows

Integrate into:

- Diagnostic reports (PDF)
- Patient timelines
- Treatment monitoring
- Research studies

---

## Support

- **Full Documentation**: [TEMPORAL_IMPORTANCE_API.md](TEMPORAL_IMPORTANCE_API.md)
- **Implementation Details**: [TEMPORAL_IMPORTANCE_IMPLEMENTATION.md](TEMPORAL_IMPORTANCE_IMPLEMENTATION.md)
- **Test Script**: `backend/scripts/test_temporal_importance.py`

---

## Quick Commands

```bash
# Run migration
psql -U db_user -d bayesian_adhd -f database/migrations/003_add_temporal_importance.sql

# Test implementation
cd backend && python scripts/test_temporal_importance.py

# Start server
cd backend && python -m app.main

# Test API (after authentication)
curl -X POST http://localhost:5000/api/temporal-importance/upload \
  -H "Cookie: session=COOKIE" \
  -F "file=@data.csv"
```

---

**Status**: ✅ Ready for Production  
**Version**: 1.0.0  
**Last Updated**: February 9, 2026
