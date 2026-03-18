# BayesianADHD

A web-based EEG analysis platform for ADHD classification using deep learning. The system processes 19-channel EEG recordings, extracts spectral features, and returns probabilistic classifications with confidence scores and clinical context.

## Features

-   **EEG Signal Processing**: Automated filtering, band power extraction, and ratio computation
-   **Deep Learning Classification**: CNN-LSTM hybrid model with Bayesian optimization
-   **Assessment Drawer Workflow**: Three-step intake (subject, upload, review) with live validation
-   **Interactive Visualizations**: Raw/filtered/band EEG previews, topographic scalp maps, and segment overlays
-   **Temporal Biomarkers**: Per-window plots and summary statistics stored per assessment
-   **Clinical Outputs**: Confidence scoring, narrative interpretation, and PDF report export
-   **Data Management**: Subject, clinician, recording, and result tracking in PostgreSQL
-   **Professional UI**: Clean, responsive interface built with Bootstrap 5

## Technology Stack

**Backend:**

-   Flask (Python web framework)
-   PyTorch (Deep learning model)
-   MNE-Python (EEG signal processing)
-   PostgreSQL 18 (Database)
-   Gunicorn (Production server)

**Frontend:**

-   HTML5/CSS3
-   Bootstrap 5.3.8
-   Vanilla JavaScript

**Infrastructure:**

-   Docker & Docker Compose
-   PostgreSQL 18

## Project Structure

```
bayesian-adhd/
├── backend/
│   ├── app/
│   │   ├── core/           # Error handling, logging, threading
│   │   ├── db/             # Database connection, repositories, schema
│   │   ├── ml/             # Model loading and inference
│   │   ├── services/       # Business logic layer
│   │   ├── static/         # CSS files
│   │   ├── templates/      # HTML templates
│   │   ├── utils/          # Signal processing utilities
│   │   └── main.py         # Flask application entry point
│   ├── notebooks/          # Jupyter notebooks for model training
│   ├── exports/            # Trained model files
│   ├── requirements.txt
│   └── Dockerfile
├── database/
│   └── init.sql            # Database initialization
└── docker-compose.yml
```

## Getting Started

### Prerequisites

-   Docker and Docker Compose
-   4GB+ RAM recommended
-   Modern web browser

### Installation

1. Clone the repository:

```bash
git clone https://github.com/makodaa/bayesian-adhd.git
cd bayesian-adhd
```

2. Start the application:

```bash
docker-compose up -d
```

3. Access the application:

-   Open your browser to `http://localhost:8000`
-   The database will be automatically initialized on first run

### Usage

1. **Start an Assessment**: Open the assessment drawer from the Dashboard
2. **Enter Subject Information**: Provide demographics and recording context
3. **Upload EEG CSV**: Validate 19-channel coverage and recording metadata
4. **Review & Submit**: Confirm details and run the assessment
5. **View Results**: Inspect classification, ratios, topographic maps, temporal plots, and EEG waveforms
6. **Export Report**: Download the PDF report for documentation

### Data Format

EEG CSV files should contain time-series data with 19 channels following the 10-20 set. The system validates channel coverage, sampling rate, and recording duration before inference.

## Development

### Local Development Setup

```bash
# Build and start in development mode (local only)
docker-compose up --build

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down

# Reset the database when problems are encountered
docker compose down -v && docker compose up --build # For macOS / Linux
docker compose down -v ; docker compose up --build # For Windows
```

### Optional Cloudflare Tunnel (Not Required for Local Dev)

Cloudflare is optional and disabled by default in development. The app runs locally at `http://localhost:8000` without a tunnel.

```bash
# Start the local stack plus Cloudflare tunnel profile
TUNNEL_TOKEN=your_token_here docker compose --profile cloudflare up --build
```

If you do not enable the `cloudflare` profile, only the local services (`database`, `backend`) are started.

### Database Access

```bash
# Connect to PostgreSQL
docker exec -it database psql -U db_user -d bayesian_adhd
```

## Model Training

The project includes Jupyter notebooks for model development:

-   `cnn_lstm_with_bo.ipynb`: CNN-LSTM training with Bayesian optimization
-   `data_preprocess.ipynb`: EEG data preprocessing pipeline
-   Additional experiments in `notebooks/old/`

## API Endpoints

-   `POST /predict` - Analyze EEG and return classification
-   `POST /api/eeg_visualization_context` - Cache uploaded EEG CSV for lazy visualization
-   `GET /api/eeg_visualizations/<context_id>/<band>.png` - Render or fetch cached EEG band preview
-   `POST /api/eeg_segment_visualization/<context_id>` - Render segment classification overlay
-   `GET /api/eeg_visualizations/<result_id>` - Retrieve stored EEG visualizations
-   `POST /api/topographic_maps` - Generate topographic maps for an uploaded EEG
-   `GET /api/topographic_maps/<result_id>` - Retrieve stored topographic maps
-   `POST /api/temporal_biomarkers` - Generate temporal biomarker plots
-   `GET /api/temporal_biomarkers/<result_id>` - Retrieve stored temporal biomarkers
-   `GET /api/model/info` - Model architecture and hyperparameter info
-   `GET /api/subjects` - List all subjects
-   `GET /api/subjects/<subject_id>` - Subject details and recordings
-   `GET /api/subjects/<subject_id>/assessments` - Subject details with assessments
-   `GET /api/clinicians` - List all clinicians
-   `GET /api/clinicians/<clinician_id>` - Clinician details and assessments
-   `GET /api/results` - List all analysis results
-   `GET /api/results/<result_id>` - Result details (band powers, ratios, metadata)
-   `GET /api/results/<result_id>/pdf` - Download PDF report

## Configuration

Key configuration in `backend/app/config.py`:

-   `SAMPLE_RATE`: EEG sampling frequency (128 Hz default)
-   `TARGET_FREQUENCY_BINS`: Number of frequency bins for model input

## Clinical Use Notice

This system is a clinical support tool and does not provide a definitive diagnosis. All findings must be interpreted by a qualified healthcare professional alongside other clinical assessments.

## License

This project is for research and educational purposes.

## Acknowledgments

Built with modern web technologies and state-of-the-art deep learning frameworks for EEG analysis.
