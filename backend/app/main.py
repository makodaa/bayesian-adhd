from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
from pathlib import Path

from .utils.timer import Timer
from .ml.model_loader import ModelLoader
from .services.file_service import FileService
from .services.eeg_service import EEGService
from .services.band_analysis_service import BandAnalysisService
from .services.visualization_service import VisualizationService
from .services.clinician_service import ClinicianService
from .services.clinician_auth_service import ClinicianAuthService
from .services.subject_service import SubjectService
from .services.recording_service import RecordingService
from .services.results_service import ResultsService
from .services.pdf_service import PDFReportService
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

# Configure session
app.secret_key = 'bayesian-adhd-secret-key-2024'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours

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
clinician_auth_service = ClinicianAuthService(clinicians_repo)
subject_service = SubjectService(subjects_repo)
recording_service = RecordingService(recordings_repo, file_service, eeg_service, band_analysis_service)
results_service = ResultsService(results_repo)
pdf_service = PDFReportService()


# Authentication decorator
def login_required(f):
    """Decorator to require login for protected routes."""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'clinician_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    
    return decorated_function


# Initialize the model on first request
@app.before_request
def initialize_model():
    """Initialize ML model before first request."""
    if not hasattr(app, 'model_initialized'):
        try:
            logger.info("Initializing model...")
            model_loader.initialize()
            logger.info("Model loaded successfully")
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

def visualize_csv_all(file):
    """
    Process CSV and visualize using Visualization Service, and return all of it
    as an array of base-64 encoded images.
    """

    logger.info(f"Visualizing CSV file: {file.filename}")
    df = file_service.read_csv(file)
    file_service.validate_eeg_data(df)

    results = []
    for band in VisualizationService.BAND_FILTERS.keys():
        with Timer() as timer:
            result = visualization_service.visualize_df(df, band)
            results.append(result)

            logger.info(f"Visualized {band} in {timer.elapsed()}")

    return visualization_service.visualize_df(df)

@error_handle
def visualize_file(file, eeg_type="raw"):
    """Route file to appropriate visualization handler"""
    if file.filename is None:
        return None

    if not file_service.is_allowed_file(file.filename):
        return {'error': 'File extension not supported'}

    return visualize_csv(file, eeg_type)

@error_handle
def visualize_file_all(file):
    """Route file to appropriate visualization handler"""
    if file.filename is None:
        return None

    if not file_service.is_allowed_file(file.filename):
        return {'error': 'File extension not supported'}

    return visualize_csv_all(file)

@app.route('/')
def home():
    """Redirect to login if not authenticated, otherwise to index."""
    if 'clinician_id' not in session:
        return redirect(url_for('login_page'))
    return redirect(url_for('index'))


@app.route('/login.html')
def login_page():
    """Render the login page."""
    if 'clinician_id' in session:
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/api/login', methods=['POST'])
def api_login():
    """API endpoint for clinician login."""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            logger.warning("Login attempt with missing credentials")
            return jsonify({'error': 'Username and password are required'}), 400
        
        # Authenticate clinician
        clinician = clinician_auth_service.authenticate(username, password)
        
        if clinician:
            # Prevent concurrent logins for the same clinician from different sessions
            try:
                already_active = clinicians_repo.is_active(clinician['id'])
            except Exception:
                already_active = False

            # If clinician already active and this request does not belong to the same session, deny login
            if already_active and session.get('clinician_id') != clinician['id']:
                logger.warning(f"Denied login for already-active clinician: {username}")
                return jsonify({'error': 'Clinician already logged in elsewhere'}), 403

            # Store clinician info in session
            session['clinician_id'] = clinician['id']
            session['clinician_name'] = clinician_auth_service.get_clinician_display_name(clinician)
            # store occupation for display in frontend
            session['clinician_occupation'] = clinician.get('occupation') if clinician.get('occupation') else ''
            session.permanent = True
            # Mark clinician as active
            clinicians_repo.set_active(clinician['id'])
            logger.info(f"Clinician logged in successfully: {username}")
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'clinician_id': clinician['id'],
                'clinician_name': session['clinician_name'],
                'clinician_occupation': session.get('clinician_occupation', '')
            }), 200
        else:
            logger.warning(f"Failed login attempt for: {username}")
            return jsonify({'error': 'Invalid username or password'}), 401
    
    except Exception as e:
        logger.error(f"Error in login endpoint: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred during login'}), 500


@app.route('/api/logout', methods=['POST'])
def api_logout():
    """API endpoint for clinician logout."""
    try:
        clinician_name = session.get('clinician_name', 'Unknown')
        clinician_id = session.get('clinician_id')
        if clinician_id:
            clinicians_repo.set_inactive(clinician_id)
        session.clear()
        logger.info(f"Clinician logged out: {clinician_name}")
        return jsonify({'success': True, 'message': 'Logged out successfully'}), 200
    except Exception as e:
        logger.error(f"Error in logout endpoint: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred during logout'}), 500


@app.route('/api/me', methods=['GET'])
def api_me():
    """Return currently authenticated clinician info from session."""
    try:
        if 'clinician_id' not in session:
            return jsonify({'error': 'not_authenticated'}), 401

        return jsonify({
            'clinician_id': session.get('clinician_id'),
            'clinician_name': session.get('clinician_name'),
            'clinician_occupation': session.get('clinician_occupation', '')
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/me: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred fetching session info'}), 500


@app.route('/logout')
def logout():
    """Logout route that redirects to login page."""
    clinician_name = session.get('clinician_name', 'Unknown')
    clinician_id = session.get('clinician_id')
    if clinician_id:
        clinicians_repo.set_inactive(clinician_id)
    session.clear()
    logger.info(f"Clinician logged out: {clinician_name}")
    return redirect(url_for('login_page'))

@app.route('/subjects.html')
@login_required
def subjects():
    return render_template('subjects.html')

@app.route('/clinicians.html')
@login_required
def clinicians():
    return render_template('clinicians.html')

@app.route('/results.html')
@login_required
def results():
    return render_template('results.html')

@app.route('/about.html')
@login_required
def about():
    return render_template('about.html')

@app.route('/index.html')
@login_required
def index():
    return render_template('index.html')

@app.route('/visualize_all_eeg', methods=['POST'])
@login_required
def visualize_all_eeg():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        result = visualize_file_all(file)
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

@app.route('/visualize_eeg/<eeg_type>', methods=['POST'])
@login_required
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
@login_required
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

        # Get or create subject (reuse existing subject if code already exists)
        logger.info(f"Getting or creating subject: {subject_code}, age={age}, gender={gender}")
        subject_id = subject_service.get_or_create_subject(subject_code, int(age), gender)

        # Use logged-in clinician from session for result creation
        clinician_id = session.get('clinician_id')
        logger.info(f"Using session clinician_id: {clinician_id}")

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
        result = eeg_service.classify_and_save(recording_id, df, clinician_id=clinician_id)

        # Compute band powers and ratios for the same recording
        logger.info(f"Computing band powers for result {result['result_id']}")
        band_powers = band_analysis_service.compute_and_save(result['result_id'], df)

        # Add band power summary to result
        result['band_analysis'] = {
            'average_absolute_power': band_powers.get('average_absolute_power', {}),
            'average_relative_power': band_powers.get('average_relative_power', {}),
            'band_ratios': band_powers.get('band_ratios', {})
        }

        logger.info(f"Classification complete: {result['classification']} ({result['confidence_score']*100:.2f}%)")
        return jsonify({
            'prediction': True,
            'result': {
                'classification': result['classification'],
                'confidence_score': result['confidence_score'],
                'subject_id': subject_id,
                'recording_id': recording_id,
                'result_id': result['result_id'],
                'clinician_id': result.get('clinician_id'),
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

@app.route('/api/clinicians/<int:clinician_id>', methods=['GET'])
def get_clinician_details(clinician_id):
    """Get clinician details with their assessment results."""
    try:
        clinician = clinicians_repo.get_with_assessments(clinician_id)
        if not clinician:
            return jsonify({'error': 'Clinician not found'}), 404
        return jsonify(clinician), 200
    except Exception as e:
        logger.error(f"Error fetching clinician details: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/clinicians', methods=['POST'])
@login_required
def create_clinician():
    """Create a new clinician (admin only)."""
    try:
        if session.get('clinician_name') != 'Admin Clinician':
            return jsonify({'error': 'Only the administrator can add clinicians'}), 403

        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing request body'}), 400

        if not data.get('password'):
            return jsonify({'error': 'Password is required for new clinicians'}), 400

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

@app.route('/api/subjects/<int:subject_id>/assessments', methods=['GET'])
def get_subject_assessments(subject_id):
    """Get subject details with their assessment results."""
    try:
        subject = subjects_repo.get_with_assessments(subject_id)
        if not subject:
            return jsonify({'error': 'Subject not found'}), 404
        return jsonify(subject), 200
    except Exception as e:
        logger.error(f"Error fetching subject assessments: {e}", exc_info=True)
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

@app.route('/api/results/<int:result_id>/pdf', methods=['GET'])
@login_required
def generate_result_pdf(result_id):
    """Generate and download PDF report for a result."""
    try:
        # Get full result details
        result = results_service.get_result_with_full_details(result_id)
        if not result:
            return jsonify({'error': 'Result not found'}), 404

        # Use clinician from result record (form-selected clinician)
        clinician_data = {
            'first_name': result.get('clinician_first_name', ''),
            'middle_name': result.get('clinician_middle_name', ''),
            'last_name': result.get('clinician_last_name', ''),
            'occupation': result.get('clinician_occupation', '')
        }

        # Generate PDF
        pdf_bytes = pdf_service.generate_report(result, clinician_data)

        # Create filename
        subject_code = result.get('subject_code', 'unknown').replace(' ', '_')
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"EEG_Report_{subject_code}_{timestamp}.pdf"

        logger.info(f"Generated PDF report for result {result_id}, size: {len(pdf_bytes)} bytes")

        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Length': str(len(pdf_bytes))
            }
        )
    except Exception as e:
        logger.error(f"Error generating PDF for result {result_id}: {e}", exc_info=True)
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500


@app.route('/api/model/info', methods=['GET'])
@login_required
def api_model_info():
    """Return model architecture and hyperparameter information."""
    import json

    try:
        # Load best parameters from JSON file
        params_path = Path(__file__).parent.parent / 'best_parameters.json'
        params = {}
        if params_path.exists():
            with open(params_path, 'r') as f:
                params = json.load(f)

        # Get model stats if model is loaded
        total_params = 0
        trainable_params = 0
        if hasattr(app, 'model_initialized') and app.model_initialized:
            try:
                model = model_loader.model
                if model is not None:
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            except Exception as e:
                logger.warning(f"Could not get model stats: {e}")

        return jsonify({
            'params': params,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_loaded': getattr(app, 'model_initialized', False)
        }), 200

    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        return jsonify({'error': 'Failed to get model info'}), 500


if __name__ == '__main__':
    import os

    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't'),
        threaded=True
    )