from flask import Flask, render_template, request, jsonify
from pathlib import Path
from .ml.model_loader import ModelLoader
from .services.file_service import FileService
from .services.eeg_service import EEGService
from .services.band_analysis_service import BandAnalysisService
from .services.visualization_service import VisualizationService
from .db.repositories.results import ResultsRepository
from .db.repositories.recordings import RecordingsRepository
from .db.repositories.band_powers import BandPowersRepository
from .db.repositories.ratios import RatiosRepository
from .db.repositories.subjects import SubjectsRepository
from .db.connection import get_db_connection
from .config import SAMPLE_RATE, TARGET_FREQUENCY_BINS
from .core.error_handle import error_handle
from .utils.mock_data import MockDataGenerator
from .core.logging_config import get_app_logger

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend

logger = get_app_logger(__name__)

# Initialize the Flask app with explicit template/static folders
_here = Path(__file__).parent
app = Flask(
    __name__,
    template_folder=str(_here / "templates"),
    static_folder=str(_here / "static"),
)

# Initialize model loader
model_loader = ModelLoader()

# Initialize repositories
results_repo = ResultsRepository()
recordings_repo = RecordingsRepository()
band_powers_repo = BandPowersRepository()
ratios_repo = RatiosRepository()
subjects_repo = SubjectsRepository()

# Initialize services
file_service = FileService()
eeg_service = EEGService(model_loader, results_repo, recordings_repo)
band_analysis_service = BandAnalysisService(band_powers_repo, ratios_repo)
visualization_service = VisualizationService()

# Initialize model on first request
@app.before_request
def initialize_model():
    """Initialize ML model before first request."""
    if not hasattr(app, 'model_initialized'):
        try:
            logger.info("Initializing model and scaler...")
            model_loader.initialize()
            logger.info("Model and scaler loaded successfully")
            app.model_initialized = True
        except FileNotFoundError as e:
            logger.error(f"Could not load model files: {e}", exc_info=True)
            logger.warning("The application will run but predictions will fail.")
            app.model_initialized = False
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            app.model_initialized = False

# Helper functions for file processing

def visualize_csv(file, eeg_type: str):
    """Process CSV and visualize using VisualizationService"""
    logger.info(f"Visualizing CSV file: {file.filename}, type: {eeg_type}")
    df = file_service.read_csv(file)
    file_service.validate_eeg_data(df)
    return visualization_service.visualize_df(df, eeg_type)

def process_csv(file):
    """Process CSV and classify using EEGService"""
    logger.info(f"Processing CSV file for classification: {file.filename}")
    df = file_service.read_csv(file)
    file_service.validate_eeg_data(df)
    
    # Create mock subject and recording for temporary solution
    logger.debug("Creating mock subject and recording")
    subject_id, recording_id = MockDataGenerator.create_mock_subject_and_recording(
        subjects_repo, recordings_repo, file.filename
    )
    
    # Classify and save results
    result = eeg_service.classify_and_save(recording_id, df)
    
    logger.info(f"Classification complete: {result['classification']} ({result['confidence_score']:.4f})")
    return {
        'classification': result['classification'],
        'confidence_score': result['confidence_score'],
        'subject_id': subject_id,
        'recording_id': recording_id,
        'result_id': result['result_id']
    }

def analyze_csv_bands(file):
    """Process CSV file and return band power analysis using BandAnalysisService"""
    logger.info(f"Analyzing band powers for file: {file.filename}")
    df = file_service.read_csv(file)
    file_service.validate_eeg_data(df)
    
    # Create mock subject and recording for temporary solution
    logger.debug("Creating mock subject and recording for band analysis")
    subject_id, recording_id = MockDataGenerator.create_mock_subject_and_recording(
        subjects_repo, recordings_repo, file.filename
    )
    
    # Create a placeholder result for band analysis
    result_id = results_repo.create_result(
        recording_id=recording_id,
        classification='Pending',
        confidence_score=0.0
    )
    
    print("Preprocessing EEG data for band analysis...")
    band_powers = band_analysis_service.compute_and_save(result_id, df)
    
    # Add metadata to response
    band_powers['metadata'] = {
        'subject_id': subject_id,
        'recording_id': recording_id,
        'result_id': result_id
    }
    
    return band_powers

@error_handle
def visualize_file(file, eeg_type="raw"):
    """Route file to appropriate visualization handler"""
    if file.filename is None:
        return None

    if not file_service.is_allowed_file(file.filename):
        return {'error': 'File extension not supported'}

    return visualize_csv(file, eeg_type)

@error_handle
def process_file(file):
    """Route file to appropriate processing handler"""
    if file.filename is None:
        return None

    if not file_service.is_allowed_file(file.filename):
        return {'error': 'File extension not supported'}

    return process_csv(file)

@app.route('/')
def home():
    return render_template('index.html')  # Render the home page with upload form

@app.route('/visualize_eeg/<eeg_type>', methods=['POST'])
def visualize_eeg(eeg_type):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        result = visualize_file(file, eeg_type)
        if result is None:
            print(f"Warning: visualize_file returned None for {file.filename}")
            return jsonify({'error': 'Something went wrong while reading the file.'}), 500
        if isinstance(result, dict) and 'error' in result:
            print(f"Warning: Error from visualize_file: {result['error']}")
            return jsonify({'error': result['error']}), 400
        return jsonify({'result': result}), 200
    except ValueError as e:
        print(f"ValueError in visualize_eeg: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Exception in visualize_eeg: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Failed to visualize: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        result = process_file(file)
        if result is None:
            return jsonify({'error': 'Something went wrong while reading the file.'}), 500
        if isinstance(result, dict) and 'error' in result:
            return jsonify({'error': result['error']}), 400
        return jsonify({'prediction': True, 'result': result}), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({'error': f'Failed to predict: {str(e)}'}), 500


@app.route('/analyze_bands', methods=['POST'])
def analyze_bands():
    """
    API endpoint to compute absolute and relative power for each frequency band.

    Returns:
    --------
    JSON with:
    - absolute_power: Power in each band for each electrode (μV²)
    - relative_power: Power as fraction of total (0-1) for each electrode
    - total_power: Total power across all bands for each electrode
    - average_absolute_power: Average across all electrodes for each band
    - average_relative_power: Average relative power across all electrodes
    - band_ratios: Clinically relevant ratios (theta/beta, etc.)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file_service.is_allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload a CSV file.'}), 400

    try:
        result = analyze_csv_bands(file)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Error in analyze_bands: {str(e)}")
        return jsonify({'error': f'Failed to analyze bands: {str(e)}'}), 500


if __name__ == '__main__':
    import os
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't'),
        threaded=True
    )