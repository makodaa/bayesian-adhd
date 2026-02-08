# BayesianADHD

A web-based EEG analysis platform for ADHD classification using deep learning. The system processes EEG recordings, extracts frequency band features, and provides probabilistic predictions with confidence scores.

## Features

-   **EEG Signal Processing**: Automated filtering and band power extraction (Delta, Theta, Alpha, Beta, Gamma)
-   **Deep Learning Classification**: CNN-LSTM hybrid model with Bayesian optimization
-   **Interactive Visualizations**: Real-time plotting of raw, filtered, and band-specific EEG signals
-   **Clinical Metrics**: Theta/Beta ratio and other clinically relevant frequency band ratios
-   **Data Management**: Subject, clinician, and recording tracking with PostgreSQL database
-   **Professional UI**: Clean, responsive interface built with Bootstrap 5

## Technology Stack

**Backend:**

-   Flask (Python web framework)
-   PyTorch (Deep learning model)
-   MNE-Python (EEG signal processing)
-   PostgreSQL (Database)
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

1. **Upload EEG Data**: Select a CSV file containing EEG recordings
2. **Enter Subject Information**: Provide demographics and recording environment details
3. **Select/Add Clinician**: Choose from existing clinicians or add a new one
4. **Analyze**: Click "Analyze EEG Recording" to process the data
5. **View Results**: Examine classification results, band power analysis, and visualizations

### Data Format

EEG CSV files should contain time-series data with columns representing different channels. The system automatically handles preprocessing and feature extraction.

## Development

### Local Development Setup

```bash
# Build and start in development mode
docker-compose up --build

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down

# Reset the database when problems are encountered
docker compose down -v && docker compose up --build # For macOS / Linux
docker compose down -v ; docker compose up --build # For Windows
```

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
-   `POST /visualize_eeg/<type>` - Generate visualization plots
-   `GET /api/subjects` - List all subjects
-   `GET /api/clinicians` - List all clinicians
-   `GET /api/results` - List all analysis results

## Configuration

Key configuration in `backend/app/config.py`:

-   `SAMPLE_RATE`: EEG sampling frequency (128 Hz default)
-   `TARGET_FREQUENCY_BINS`: Number of frequency bins for model input

## License

This project is for research and educational purposes.

## Acknowledgments

Built with modern web technologies and state-of-the-art deep learning frameworks for EEG analysis.
