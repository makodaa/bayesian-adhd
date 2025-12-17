from flask import Flask, render_template, request, jsonify
from pathlib import Path
from .ml.model_loader import ModelLoader
from .services.file_service import FileService
from .services.eeg_service import EEGService
from .services.band_analysis_service import BandAnalysisService
from .services.visualization_service import VisualizationService
from .services.clinician_service import ClinicianService
from .services.subject_service import SubjectService
from .services.recording_service import RecordingService
from .services.results_service import ResultsService
from .db.repositories.results import ResultsRepository
from .db.repositories.recordings import RecordingsRepository
from .db.repositories.band_powers import BandPowersRepository
from .db.repositories.ratios import RatiosRepository
from .db.repositories.subjects import SubjectsRepository
from .db.repositories.clinicians import CliniciansRepository
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
clinicians_repo = CliniciansRepository()

# Initialize services
file_service = FileService()
eeg_service = EEGService(model_loader, results_repo, recordings_repo)
band_analysis_service = BandAnalysisService(band_powers_repo, ratios_repo)
visualization_service = VisualizationService()
clinician_service = ClinicianService(clinicians_repo)
subject_service = SubjectService(subjects_repo)
recording_service = RecordingService(recordings_repo, file_service, eeg_service, band_analysis_service)
results_service = ResultsService(results_repo)

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

@error_handle
def visualize_file(file, eeg_type="raw"):
    """Route file to appropriate visualization handler"""
    if file.filename is None:
        return None

    if not file_service.is_allowed_file(file.filename):
        return {'error': 'File extension not supported'}

    return visualize_csv(file, eeg_type)

@app.route('/')
def home():
    return render_template('index.html')  # Render the home page with upload form

@app.route('/subjects.html')
def subjects():
    return render_template('subjects.html')

@app.route('/clinicians.html')
def clinicians():
    return render_template('clinicians.html')

@app.route('/results.html')
def results():
    return render_template('results.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

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
        # Extract form data
        subject_code = request.form.get('subject_code')
        age = request.form.get('age')
        gender = request.form.get('gender')
        clinician_name = request.form.get('clinician_name')
        occupation = request.form.get('occupation')
        sleep_hours = request.form.get('sleep_hours')
        food_intake = request.form.get('food_intake')
        caffeinated = request.form.get('caffeinated')
        medicated = request.form.get('medicated')
        medication_intake = request.form.get('medication_intake')
        notes = request.form.get('notes')
        
        # Validate required subject data
        if not subject_code or not age or not gender:
            logger.error("Missing required subject data")
            return jsonify({'error': 'Subject code, age, and gender are required'}), 400
        
        # Process file
        df = file_service.read_csv(file)
        file_service.validate_eeg_data(df)
        
        # Create subject
        logger.info(f"Creating subject: {subject_code}, age={age}, gender={gender}")
        subject_id = subject_service.create_subject(subject_code, int(age), gender)
        
        # Create recording with environmental data
        logger.info(f"Creating recording for subject {subject_id}")
        recording_id = recording_service.create_recording(
            subject_id=subject_id,
            file_name=file.filename,
            sleep_hours=float(sleep_hours) if sleep_hours else None,
            food_intake=food_intake,
            caffeinated=caffeinated == 'true' if caffeinated else None,
            medicated=medicated == 'true' if medicated else None,
            medication_intake=medication_intake,
            notes=notes
        )
        
        # Classify and save results
        result = eeg_service.classify_and_save(recording_id, df)
        
        # Compute band powers and ratios for the same recording
        logger.info(f"Computing band powers for result {result['result_id']}")
        band_powers = band_analysis_service.compute_and_save(result['result_id'], df)
        
        # Add band power summary to result
        result['band_analysis'] = {
            'average_absolute_power': band_powers.get('average_absolute_power', {}),
            'average_relative_power': band_powers.get('average_relative_power', {}),
            'band_ratios': band_powers.get('band_ratios', {})
        }
        
        # Store clinician info if provided (for future report generation)
        if clinician_name:
            logger.info(f"Clinician info provided: {clinician_name}, {occupation}")
        
        logger.info(f"Classification complete: {result['classification']} ({result['confidence_score']:.4f})")
        return jsonify({
            'prediction': True,
            'result': {
                'classification': result['classification'],
                'confidence_score': result['confidence_score'],
                'subject_id': subject_id,
                'recording_id': recording_id,
                'result_id': result['result_id'],
                'band_analysis': result.get('band_analysis', {})
            }
        }), 200
    except ValueError as e:
        logger.error(f"Validation error in predict: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in predict: {e}", exc_info=True)
        return jsonify({'error': f'Failed to predict: {str(e)}'}), 500


# API Endpoints for Management Pages

@app.route('/api/clinicians', methods=['GET'])
def get_clinicians():
    """Get all clinicians for datalist autocomplete."""
    try:
        clinicians = clinician_service.get_all_clinicians()
        formatted = clinician_service.format_clinicians_for_frontend(clinicians)
        return jsonify(formatted), 200
    except Exception as e:
        logger.error(f"Error fetching clinicians: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/clinicians', methods=['POST'])
def create_clinician():
    """Create a new clinician."""
    try:
        data = request.get_json()
        clinician_id = clinician_service.create_clinician(data)
        return jsonify({'id': clinician_id, 'message': 'Clinician created successfully'}), 201
    except Exception as e:
        logger.error(f"Error creating clinician: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/subjects', methods=['GET'])
def get_subjects():
    """Get all subjects."""
    try:
        subjects = subject_service.get_all_subjects()
        # Add recording count for each subject
        for subject in subjects:
            recordings = recording_service.get_recordings_by_subject(subject['id'])
            subject['recording_count'] = len(recordings)
        return jsonify(subjects), 200
    except Exception as e:
        logger.error(f"Error fetching subjects: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/subjects/<int:subject_id>', methods=['GET'])
def get_subject(subject_id):
    """Get a specific subject with their recordings."""
    try:
        subject = subject_service.get_subject(subject_id)
        if not subject:
            return jsonify({'error': 'Subject not found'}), 404
        recordings = recording_service.get_recordings_by_subject(subject_id)
        subject['recordings'] = recordings
        return jsonify(subject), 200
    except Exception as e:
        logger.error(f"Error fetching subject {subject_id}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get all analysis results with subject and recording details."""
    try:
        results = results_service.get_all_results_with_details()
        return jsonify(results), 200
    except Exception as e:
        logger.error(f"Error fetching results: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<int:result_id>', methods=['GET'])
def get_result(result_id):
    """Get detailed result including band powers and ratios."""
    try:
        result = results_service.get_result_with_full_details(result_id)
        if not result:
            return jsonify({'error': 'Result not found'}), 404
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error fetching result {result_id}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't'),
        threaded=True
    )