from os import PathLike
from pathlib import Path
from typing import cast

import matplotlib
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from .core.logging_config import get_app_logger
from .db.repositories.band_powers import BandPowersRepository
from .db.repositories.clinicians import CliniciansRepository
from .db.repositories.ratios import RatiosRepository
from .db.repositories.recordings import RecordingsRepository
from .db.repositories.results import ResultsRepository
from .db.repositories.subjects import SubjectsRepository
from .db.repositories.temporal_plots import TemporalPlotsRepository
from .db.repositories.temporal_summaries import TemporalSummariesRepository
from .db.repositories.topographic_maps import TopographicMapsRepository
from .db.repositories.eeg_visualizations import EEGVisualizationsRepository
from .ml.model_loader import ModelLoader
from .services.band_analysis_service import BandAnalysisService
from .services.acos_service import AcosService
from .services.clinician_auth_service import ClinicianAuthService
from .services.clinician_service import ClinicianService
from .services.eeg_service import EEGService
from .services.file_service import FileService
from .services.pdf_service import PDFReportService
from .services.recording_service import RecordingService
from .services.results_service import ResultsService
from .services.subject_service import SubjectService
from .services.temporal_biomarker_service import TemporalBiomarkerService
from .services.topographic_service import TopographicService
from .services.visualization_service import BandFilter, VisualizationService
from .services.visualization_cache_service import (
    ContextAccessError,
    ContextExpiredError,
    ContextNotFoundError,
    VisualizationCacheService,
)

matplotlib.use("Agg")  # Non-GUI backend

logger = get_app_logger(__name__)

# Initialize the Flask app with explicit template/static folders
_here = Path(__file__).parent


class Application(Flask):
    def __init__(
        self,
        import_name: str,
        static_url_path: str | None = None,
        static_folder: str | PathLike[str] | None = "static",
        static_host: str | None = None,
        host_matching: bool = False,
        subdomain_matching: bool = False,
        template_folder: str | PathLike[str] | None = "templates",
        instance_path: str | None = None,
        instance_relative_config: bool = False,
        root_path: str | None = None,
    ):
        super().__init__(
            import_name,
            static_url_path=static_url_path,
            static_folder=static_folder,
            static_host=static_host,
            host_matching=host_matching,
            subdomain_matching=subdomain_matching,
            template_folder=template_folder,
            instance_path=instance_path,
            instance_relative_config=instance_relative_config,
            root_path=root_path,
        )
        self.model_initialized = False


app = Application(
    __name__,
    template_folder=str(_here / "templates"),
    static_folder=str(_here / "static"),
)

# Configure session
app.secret_key = "bayesian-adhd-secret-key-2024"
app.config["SESSION_TYPE"] = "filesystem"
app.config["PERMANENT_SESSION_LIFETIME"] = 86400  # 24 hours

# Initialize model loader
model_loader = ModelLoader()

# Initialize repositories
results_repo = ResultsRepository()
recordings_repo = RecordingsRepository()
band_powers_repo = BandPowersRepository()
ratios_repo = RatiosRepository()
subjects_repo = SubjectsRepository()
clinicians_repo = CliniciansRepository()
temporal_plots_repo = TemporalPlotsRepository()
temporal_summaries_repo = TemporalSummariesRepository()
topographic_maps_repo = TopographicMapsRepository()
eeg_visualizations_repo = EEGVisualizationsRepository()

# Initialize services
file_service = FileService()
eeg_service = EEGService(model_loader, results_repo, recordings_repo)
band_analysis_service = BandAnalysisService(band_powers_repo, ratios_repo)
acos_service = AcosService()
visualization_service = VisualizationService()
visualization_cache_service = VisualizationCacheService()
topographic_service = TopographicService()
temporal_biomarker_service = TemporalBiomarkerService()
clinician_service = ClinicianService(clinicians_repo)
clinician_auth_service = ClinicianAuthService(clinicians_repo)
subject_service = SubjectService(subjects_repo)
recording_service = RecordingService(
    recordings_repo, file_service, eeg_service, band_analysis_service
)
results_service = ResultsService(results_repo)
pdf_service = PDFReportService()


# Authentication decorator
def login_required(f):
    """Decorator to require login for protected routes."""
    from functools import wraps

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "clinician_id" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)

    return decorated_function


# Initialize the model on first request
@app.before_request
def initialize_model():
    """Initialize ML model before first request."""
    if not hasattr(app, "model_initialized") or not app.model_initialized:
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


@app.route("/")
def home():
    """Redirect to login if not authenticated, otherwise to index."""
    if "clinician_id" not in session:
        return redirect(url_for("login_page"))
    return redirect(url_for("index"))


@app.route("/login.html")
def login_page():
    """Render the login page."""
    if "clinician_id" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    """API endpoint for clinician login."""
    try:
        data = request.get_json()
        username = data.get("username", "").strip()
        password = data.get("password", "")

        if not username or not password:
            logger.warning("Login attempt with missing credentials")
            return jsonify({"error": "Username and password are required"}), 400

        # Authenticate clinician
        clinician = clinician_auth_service.authenticate(username, password)

        if clinician:
            # Prevent concurrent logins for the same clinician from different sessions
            try:
                already_active = clinicians_repo.is_active(clinician["id"])
            except Exception:
                already_active = False

            # If clinician already active and this request does not belong to the same session, deny login
            if already_active and session.get("clinician_id") != clinician["id"]:
                logger.warning(f"Denied login for already-active clinician: {username}")
                return jsonify({"error": "Clinician already logged in elsewhere"}), 403

            # Store clinician info in session
            session["clinician_id"] = clinician["id"]
            session["clinician_name"] = (
                clinician_auth_service.get_clinician_display_name(clinician)
            )
            # store occupation for display in frontend
            session["clinician_occupation"] = (
                clinician.get("occupation") if clinician.get("occupation") else ""
            )
            session.permanent = True
            # Mark clinician as active
            clinicians_repo.set_active(clinician["id"])
            logger.info(f"Clinician logged in successfully: {username}")
            return jsonify(
                {
                    "success": True,
                    "message": "Login successful",
                    "clinician_id": clinician["id"],
                    "clinician_name": session["clinician_name"],
                    "clinician_occupation": session.get("clinician_occupation", ""),
                }
            ), 200
        else:
            logger.warning(f"Failed login attempt for: {username}")
            return jsonify({"error": "Invalid username or password"}), 401

    except Exception as e:
        logger.error(f"Error in login endpoint: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during login"}), 500


@app.route("/api/logout", methods=["POST"])
def api_logout():
    """API endpoint for clinician logout."""
    try:
        clinician_name = session.get("clinician_name", "Unknown")
        clinician_id = session.get("clinician_id")
        if clinician_id:
            clinicians_repo.set_inactive(clinician_id)
        session.clear()
        logger.info(f"Clinician logged out: {clinician_name}")
        return jsonify({"success": True, "message": "Logged out successfully"}), 200
    except Exception as e:
        logger.error(f"Error in logout endpoint: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during logout"}), 500


@app.route("/api/me", methods=["GET"])
def api_me():
    """Return currently authenticated clinician info from session."""
    try:
        if "clinician_id" not in session:
            return jsonify({"error": "not_authenticated"}), 401

        return jsonify(
            {
                "clinician_id": session.get("clinician_id"),
                "clinician_name": session.get("clinician_name"),
                "clinician_occupation": session.get("clinician_occupation", ""),
            }
        ), 200
    except Exception as e:
        logger.error(f"Error in /api/me: {e}", exc_info=True)
        return jsonify({"error": "An error occurred fetching session info"}), 500


@app.route("/logout")
def logout():
    """Logout route that redirects to login page."""
    clinician_name = session.get("clinician_name", "Unknown")
    clinician_id = session.get("clinician_id")
    if clinician_id:
        clinicians_repo.set_inactive(clinician_id)
    session.clear()
    logger.info(f"Clinician logged out: {clinician_name}")
    return redirect(url_for("login_page"))


@app.route("/subjects.html")
@login_required
def subjects():
    return render_template("subjects.html")


@app.route("/clinicians.html")
@login_required
def clinicians():
    return render_template("clinicians.html")


@app.route("/results.html")
@login_required
def results():
    return render_template("results.html")


@app.route("/about.html")
@login_required
def about():
    return render_template("about.html")


@app.route("/index.html")
@login_required
def index():
    return render_template("index.html")


@app.route("/api/eeg_visualization_context", methods=["POST"])
@login_required
def create_eeg_visualization_context():
    """Create a cached visualization context for an uploaded EEG file."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    filename = file.filename
    if filename is None or filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file_service.is_allowed_file(filename):
        return jsonify({"error": "File extension not supported"}), 400

    clinician_id = session.get("clinician_id")
    if not isinstance(clinician_id, int):
        return jsonify({"error": "Invalid clinician session"}), 401

    try:
        logger.info(
            f"Creating visualization context for clinician {clinician_id}, file={filename}"
        )
        file_bytes = file.stream.read()
        df = file_service.read_csv_bytes(file_bytes, filename)
        file_service.validate_eeg_data(df)

        context = visualization_cache_service.create_or_refresh_context(
            file_bytes=file_bytes,
            clinician_id=clinician_id,
            filename=filename,
        )

        logger.info(
            "Visualization context ready: "
            f"context_id={context['context_id']}, created={context['created']}"
        )

        return jsonify(
            {
                "context_id": context["context_id"],
                "bands": list(VisualizationService.BAND_FILTERS.keys()),
                "expires_at": context["expires_at"],
                "created": context["created"],
            }
        ), 200
    except ValueError as e:
        logger.warning(f"Visualization context validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error creating visualization context: {e}", exc_info=True)
        return jsonify({"error": "Failed to create visualization context"}), 500


@app.route("/api/eeg_visualizations/<context_id>/<band>.png", methods=["GET"])
@login_required
def get_eeg_visualization_image(context_id: str, band: str):
    """Return a cached PNG for one EEG band. Supports ?quality=preview|detail."""
    if band not in VisualizationService.BAND_FILTERS:
        return jsonify({"error": "Invalid EEG type"}), 400

    quality = request.args.get("quality", "preview")
    if quality not in ("preview", "detail"):
        quality = "preview"

    clinician_id = session.get("clinician_id")
    if not isinstance(clinician_id, int):
        return jsonify({"error": "Invalid clinician session"}), 401

    try:
        logger.info(
            f"Visualization request: clinician={clinician_id}, context={context_id}, "
            f"band={band}, quality={quality}"
        )
        cached_path = visualization_cache_service.get_cached_image_path(
            context_id=context_id,
            clinician_id=clinician_id,
            band=band,
            quality=quality,
        )
        if cached_path is not None:
            logger.info(
                f"Visualization cache HIT for context={context_id}, band={band}, quality={quality}"
            )
            response = send_file(cached_path, mimetype="image/png", conditional=True)
            response.headers["X-Visualization-Cache"] = "HIT"
            response.headers["X-Visualization-Quality"] = quality
            response.headers["Cache-Control"] = "private, max-age=3600"
            return response

        csv_bytes = visualization_cache_service.read_context_csv(
            context_id=context_id,
            clinician_id=clinician_id,
        )
        df = file_service.read_csv_bytes(csv_bytes, f"{context_id}.csv")
        file_service.validate_eeg_data(df)

        if quality == "detail":
            image_bytes = visualization_service.render_detail_png(
                df,
                cast(BandFilter, band),
            )
        else:
            image_bytes = visualization_service.render_preview_png(
                df,
                cast(BandFilter, band),
            )

        image_path = visualization_cache_service.store_image(
            context_id=context_id,
            clinician_id=clinician_id,
            band=band,
            image_bytes=image_bytes,
            quality=quality,
        )

        logger.info(
            f"Visualization cache MISS for context={context_id}, band={band}, quality={quality}"
        )

        response = send_file(image_path, mimetype="image/png", conditional=True)
        response.headers["X-Visualization-Cache"] = "MISS"
        response.headers["X-Visualization-Quality"] = quality
        response.headers["Cache-Control"] = "private, max-age=3600"
        return response
    except ContextAccessError:
        return jsonify({"error": "Not authorized for this visualization context"}), 403
    except (ContextNotFoundError, ContextExpiredError) as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating visualization image: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate visualization"}), 500


@app.route("/api/eeg_segment_visualization/<context_id>", methods=["POST"])
@login_required
def get_eeg_segment_visualization(context_id: str):
    """Render a segment classification overlay visualization.

    Expects JSON body with ``window_predictions`` (list of per-window dicts)
    and optional ``quality`` ("preview" | "detail").
    """
    clinician_id = session.get("clinician_id")
    if not isinstance(clinician_id, int):
        return jsonify({"error": "Invalid clinician session"}), 401

    body = request.get_json(silent=True) or {}
    window_predictions = body.get("window_predictions")
    if not window_predictions or not isinstance(window_predictions, list):
        return jsonify({"error": "window_predictions array is required"}), 400

    quality = body.get("quality", "preview")
    if quality not in ("preview", "detail"):
        quality = "preview"

    try:
        # Check cache for previously rendered segment image
        band_key = "segments"
        cached_path = visualization_cache_service.get_cached_image_path(
            context_id=context_id,
            clinician_id=clinician_id,
            band=band_key,
            quality=quality,
        )
        if cached_path is not None:
            response = send_file(cached_path, mimetype="image/png", conditional=True)
            response.headers["X-Visualization-Cache"] = "HIT"
            response.headers["X-Visualization-Quality"] = quality
            return response

        csv_bytes = visualization_cache_service.read_context_csv(
            context_id=context_id,
            clinician_id=clinician_id,
        )
        df = file_service.read_csv_bytes(csv_bytes, f"{context_id}.csv")
        file_service.validate_eeg_data(df)

        image_bytes = visualization_service.render_segment_png(
            df, window_predictions, quality=quality,
        )

        image_path = visualization_cache_service.store_image(
            context_id=context_id,
            clinician_id=clinician_id,
            band=band_key,
            image_bytes=image_bytes,
            quality=quality,
        )

        response = send_file(image_path, mimetype="image/png", conditional=True)
        response.headers["X-Visualization-Cache"] = "MISS"
        response.headers["X-Visualization-Quality"] = quality
        return response
    except ContextAccessError:
        return jsonify({"error": "Not authorized for this visualization context"}), 403
    except (ContextNotFoundError, ContextExpiredError) as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating segment visualization: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate segment visualization"}), 500


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    filename: str | None = file.filename
    if filename is None or filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Extract form data
        subject_code = request.form.get("subject_code")
        age = request.form.get("age")
        gender = request.form.get("gender")

        # Unused variables
        # clinician_name = request.form.get("clinician_name")
        # occupation = request.form.get("occupation")

        sleep_hours = request.form.get("sleep_hours")
        food_intake = request.form.get("food_intake")
        caffeinated = request.form.get("caffeinated")
        medicated = request.form.get("medicated")
        medication_intake = request.form.get("medication_intake")
        notes = request.form.get("notes")

        acos_item_scores: dict[str, int] | None = None
        acos_result: dict[str, object] | None = None
        acos_raw: dict[int, int] = {}
        for item_index in range(1, 16):
            raw_value = request.form.get(f"acos_item_{item_index}")
            if raw_value is not None and raw_value != "":
                try:
                    score_value = int(raw_value)
                except ValueError as exc:
                    raise ValueError(f"ACOS item {item_index} must be an integer") from exc
                if not (0 <= score_value <= 5):
                    raise ValueError(f"ACOS item {item_index} must be between 0 and 5")
                acos_raw[item_index] = score_value

        if acos_raw:
            if len(acos_raw) != 15:
                raise ValueError("All 15 ACOS-C items are required when ACOS scoring is provided")
            acos_result = dict(acos_service.compute(acos_raw))
            acos_item_scores = {
                f"item_{index}": value for index, value in sorted(acos_raw.items())
            }

        # Validate required subject data
        if not subject_code or not age or not gender:
            logger.error(
                f"Missing required subject data: {subject_code}, {age}, {gender}"
            )
            return jsonify({"error": "Subject code, age, and gender are required"}), 400

        # Validate optional numeric fields
        if sleep_hours is not None and sleep_hours != "":
            try:
                sleep_hours_float = float(sleep_hours)
                if not (0 <= sleep_hours_float <= 99.99):
                    return jsonify({"error": "Sleep hours must be between 0 and 99.99"}), 400
                sleep_hours = str(sleep_hours_float)
            except ValueError:
                return jsonify({"error": "Invalid value for sleep hours"}), 400

        # Use logged-in clinician from session for result creation
        clinician_id = session.get("clinician_id")
        if not isinstance(clinician_id, int):
            raise ValueError("Invalid clinician_id")
        logger.info(f"Using session clinician_id: {clinician_id}")

        # Process file once and keep bytes for visualization cache reuse
        file_bytes = file.stream.read()
        df = file_service.read_csv_bytes(file_bytes, filename)
        file_service.validate_eeg_data(df)

        visualization_context_id: str | None = None
        try:
            viz_context = visualization_cache_service.create_or_refresh_context(
                file_bytes=file_bytes,
                clinician_id=clinician_id,
                filename=filename,
            )
            visualization_context_id = viz_context["context_id"]
            logger.info(
                f"Predict linked visualization context: {visualization_context_id}"
            )
        except Exception as exc:
            logger.warning(
                f"Could not create visualization context during predict: {exc}",
                exc_info=True,
            )

        # Get or create subject (reuse existing subject if code already exists)
        logger.info(
            f"Getting or creating subject: {subject_code}, age={age}, gender={gender}"
        )
        subject_id = subject_service.get_or_create_subject(
            subject_code, int(age), gender
        )

        # Create recording with environmental data
        logger.info(f"Creating recording for subject {subject_id}")
        recording_id = recording_service.create_recording(
            subject_id=subject_id,
            file_name=filename,
            sleep_hours=float(sleep_hours) if sleep_hours else None,
            food_intake=food_intake,
            caffeinated=caffeinated == "true" if caffeinated else None,
            medicated=medicated == "true" if medicated else None,
            medication_intake=medication_intake,
            notes=notes,
        )

        # Classify and save results
        result = eeg_service.classify_and_save(
            recording_id,
            df,
            clinician_id=clinician_id,
            acos_result=acos_result,
            acos_item_scores=acos_item_scores,
        )

        # Compute band powers and ratios for the same recording
        logger.info(f"Computing band powers for result {result['result_id']}")
        band_powers = band_analysis_service.compute_and_save(result["result_id"], df)

        # Generate and save topographic maps
        logger.info(f"Generating topographic maps for result {result['result_id']}")
        try:
            topo_data = topographic_service.generate_all_topomaps(df)
            # Save absolute power maps
            for band, image in topo_data.get("absolute_power_maps", {}).items():
                topographic_maps_repo.create_map(result["result_id"], "absolute", band, image)
            # Save relative power maps
            for band, image in topo_data.get("relative_power_maps", {}).items():
                topographic_maps_repo.create_map(result["result_id"], "relative", band, image)
            # Save TBR map
            if topo_data.get("tbr_map"):
                topographic_maps_repo.create_map(result["result_id"], "tbr", None, topo_data["tbr_map"])
            logger.info(f"Topographic maps saved for result {result['result_id']}")
        except Exception as e:
            logger.warning(f"Failed to generate/save topographic maps: {e}", exc_info=True)

        # Generate and save temporal biomarker plots
        logger.info(f"Computing temporal biomarkers for result {result['result_id']}")
        try:
            temporal_data = temporal_biomarker_service.generate_temporal_plots(df)
            # Save plot images
            for plot in temporal_data.get("plots", []):
                temporal_plots_repo.create_plot(result["result_id"], plot["group"], plot["image"])
            # Save summary statistics
            for key, stats in temporal_data.get("summary", {}).items():
                temporal_summaries_repo.create_summary(
                    result["result_id"], key,
                    stats["mean"], stats["std"], stats["min"], stats["max"]
                )
            logger.info(f"Temporal biomarkers saved for result {result['result_id']}")
        except Exception as e:
            logger.warning(f"Failed to generate/save temporal biomarkers: {e}", exc_info=True)

        # Generate and persist EEG visualization images to database
        logger.info(f"Generating EEG visualizations for result {result['result_id']}")
        try:
            import base64

            viz_bands = ["raw", "filtered", "delta", "theta", "alpha", "beta", "gamma"]
            for band_name in viz_bands:
                img_bytes = visualization_service.render_detail_png(df, band_name)
                b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode("ascii")
                eeg_visualizations_repo.create_visualization(
                    result["result_id"], band_name, b64
                )
            logger.info(f"Saved {len(viz_bands)} band visualizations for result {result['result_id']}")

            # Segment visualization (requires window_predictions)
            window_preds = result.get("window_predictions", [])
            if window_preds:
                seg_bytes = visualization_service.render_segment_png(
                    df, window_preds, quality="detail"
                )
                seg_b64 = "data:image/png;base64," + base64.b64encode(seg_bytes).decode("ascii")
                eeg_visualizations_repo.create_visualization(
                    result["result_id"], "segments", seg_b64
                )
                logger.info(f"Saved segment visualization for result {result['result_id']}")
        except Exception as e:
            logger.warning(f"Failed to generate/save EEG visualizations: {e}", exc_info=True)

        # Add band power summary to result
        result["band_analysis"] = {
            "average_absolute_power": band_powers.get("average_absolute_power", {}),
            "average_relative_power": band_powers.get("average_relative_power", {}),
            "absolute_power": band_powers.get("absolute_power", {}),
            "relative_power": band_powers.get("relative_power", {}),
            "band_ratios": band_powers.get("band_ratios", {}),
        }

        logger.info(
            f"Classification complete: {result['classification']} ({result['confidence_score'] * 100:.2f}%)"
        )
        return jsonify(
            {
                "prediction": True,
                "result": {
                    "classification": result["classification"],
                    "confidence_score": result["confidence_score"],
                    "subject_id": subject_id,
                    "recording_id": recording_id,
                    "result_id": result["result_id"],
                    "clinician_id": result.get("clinician_id"),
                    "band_analysis": result.get("band_analysis", {}),
                    "visualization_context_id": visualization_context_id,
                    "window_predictions": result.get("window_predictions", []),
                    "acos": result.get("acos"),
                },
            }
        ), 200
    except ValueError as e:
        logger.error(f"Validation error in predict: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in predict: {e}", exc_info=True)
        return jsonify({"error": f"Failed to predict: {str(e)}"}), 500


# API Endpoints for Management Pages


@app.route("/api/clinicians", methods=["GET"])
def get_clinicians():
    """Get all clinicians for datalist autocomplete."""
    try:
        clinicians = clinician_service.get_all_clinicians()
        formatted = clinician_service.format_clinicians_for_frontend(clinicians)
        return jsonify(formatted), 200
    except Exception as e:
        logger.error(f"Error fetching clinicians: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/clinicians/<int:clinician_id>", methods=["GET"])
def get_clinician_details(clinician_id):
    """Get clinician details with their assessment results."""
    try:
        clinician = clinicians_repo.get_with_assessments(clinician_id)
        if not clinician:
            return jsonify({"error": "Clinician not found"}), 404
        return jsonify(clinician), 200
    except Exception as e:
        logger.error(f"Error fetching clinician details: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/clinicians", methods=["POST"])
@login_required
def create_clinician():
    """Create a new clinician (admin only)."""
    try:
        if session.get("clinician_name") != "Admin Clinician":
            return jsonify({"error": "Only the administrator can add clinicians"}), 403

        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        if not data.get("password"):
            return jsonify({"error": "Password is required for new clinicians"}), 400

        clinician_id = clinician_service.create_clinician(data)
        return jsonify(
            {"id": clinician_id, "message": "Clinician created successfully"}
        ), 201
    except Exception as e:
        logger.error(f"Error creating clinician: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/subjects", methods=["GET"])
def get_subjects():
    """Get all subjects."""
    try:
        subjects = subject_service.get_all_subjects()
        # Add recording count for each subject
        for subject in subjects:
            recordings = recording_service.get_recordings_by_subject(subject["id"])
            subject["recording_count"] = len(recordings)
        return jsonify(subjects), 200
    except Exception as e:
        logger.error(f"Error fetching subjects: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/subjects/<int:subject_id>", methods=["GET"])
def get_subject(subject_id):
    """Get a specific subject with their recordings."""
    try:
        subject = subject_service.get_subject(subject_id)
        if not subject:
            return jsonify({"error": "Subject not found"}), 404
        recordings = recording_service.get_recordings_by_subject(subject_id)
        subject["recordings"] = recordings
        return jsonify(subject), 200
    except Exception as e:
        logger.error(f"Error fetching subject {subject_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/subjects/<int:subject_id>/assessments", methods=["GET"])
def get_subject_assessments(subject_id):
    """Get subject details with their assessment results."""
    try:
        subject = subjects_repo.get_with_assessments(subject_id)
        if not subject:
            return jsonify({"error": "Subject not found"}), 404
        return jsonify(subject), 200
    except Exception as e:
        logger.error(f"Error fetching subject assessments: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/results", methods=["GET"])
def get_results():
    """Get all analysis results with subject and recording details."""
    try:
        results = results_service.get_all_results_with_details()
        return jsonify(results), 200
    except Exception as e:
        logger.error(f"Error fetching results: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/<int:result_id>", methods=["GET"])
def get_result(result_id):
    """Get detailed result including band powers and ratios."""
    try:
        result = results_service.get_result_with_full_details(result_id)
        if not result:
            return jsonify({"error": "Result not found"}), 404
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error fetching result {result_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/<int:result_id>/pdf", methods=["GET"])
@login_required
def generate_result_pdf(result_id):
    """Generate and download PDF report for a result."""
    try:
        # Get full result details
        result = results_service.get_result_with_full_details(result_id)
        if not result:
            return jsonify({"error": "Result not found"}), 404

        # Use clinician from result record (form-selected clinician)
        clinician_data = {
            "first_name": result.get("clinician_first_name", ""),
            "middle_name": result.get("clinician_middle_name", ""),
            "last_name": result.get("clinician_last_name", ""),
            "occupation": result.get("clinician_occupation", ""),
        }

        # Generate PDF with EEG visualizations
        eeg_viz_rows = eeg_visualizations_repo.get_by_result(result_id)
        eeg_visualizations = {}
        for row in eeg_viz_rows:
            eeg_visualizations[row["band_name"]] = row["image_data"]
        result["eeg_visualizations"] = eeg_visualizations

        pdf_bytes = pdf_service.generate_report(result, clinician_data)

        # Create filename
        subject_code = result.get("subject_code", "unknown").replace(" ", "_")
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"EEG_Report_{subject_code}_{timestamp}.pdf"

        logger.info(
            f"Generated PDF report for result {result_id}, size: {len(pdf_bytes)} bytes"
        )

        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            },
        )
    except Exception as e:
        logger.error(f"Error generating PDF for result {result_id}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500


@app.route("/api/eeg_visualizations/<int:result_id>", methods=["GET"])
@login_required
def get_eeg_visualizations_by_result(result_id):
    """Retrieve stored EEG visualization images for a result."""
    try:
        rows = eeg_visualizations_repo.get_by_result(result_id)
        if not rows:
            return jsonify({"error": "No visualizations found for this result"}), 404

        visualizations = {}
        for row in rows:
            visualizations[row["band_name"]] = row["image_data"]

        return jsonify({"result_id": result_id, "visualizations": visualizations}), 200
    except Exception as e:
        logger.error(
            f"Error fetching EEG visualizations for result {result_id}: {e}",
            exc_info=True,
        )
        return jsonify({"error": str(e)}), 500


@app.route("/api/topographic_maps", methods=["POST"])
@login_required
def api_topographic_maps():
    """Generate topographic scalp heatmaps from an uploaded EEG CSV file.

    Returns absolute power, relative power topomaps per band,
    plus a Theta/Beta Ratio (TBR) topomap.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename is None or file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = file_service.read_csv(file)
        file_service.validate_eeg_data(df)
        topo_data = topographic_service.generate_all_topomaps(df)

        return jsonify({
            "absolute_power_maps": topo_data["absolute_power_maps"],
            "relative_power_maps": topo_data["relative_power_maps"],
            "tbr_map": topo_data["tbr_map"],
        }), 200
    except ValueError as e:
        logger.error(f"Validation error in topographic maps: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating topographic maps: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate topographic maps: {str(e)}"}), 500


@app.route("/api/topographic_maps/<int:result_id>", methods=["GET"])
@login_required
def api_topographic_maps_by_result(result_id):
    """Retrieve stored topographic scalp heatmaps for a result.

    First tries to load pre-generated images from the database.
    Falls back to generating on-the-fly from stored band powers.
    """
    try:
        # Try stored images first
        stored_maps = topographic_maps_repo.get_by_result(result_id)
        if stored_maps:
            absolute_power_maps = {}
            relative_power_maps = {}
            tbr_map = None
            for row in stored_maps:
                if row["map_type"] == "absolute":
                    absolute_power_maps[row["band"]] = row["map_image"]
                elif row["map_type"] == "relative":
                    relative_power_maps[row["band"]] = row["map_image"]
                elif row["map_type"] == "tbr":
                    tbr_map = row["map_image"]
            return jsonify({
                "absolute_power_maps": absolute_power_maps,
                "relative_power_maps": relative_power_maps,
                "tbr_map": tbr_map,
            }), 200

        # Fallback: generate from stored band powers
        band_powers = band_powers_repo.get_by_result(result_id)
        if not band_powers:
            return jsonify({"error": "No band power data found for this result"}), 404

        topo_data = topographic_service.generate_topomaps_from_db(band_powers)

        return jsonify({
            "absolute_power_maps": topo_data["absolute_power_maps"],
            "relative_power_maps": topo_data["relative_power_maps"],
            "tbr_map": topo_data["tbr_map"],
        }), 200
    except Exception as e:
        logger.error(f"Error generating topographic maps for result {result_id}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate topographic maps: {str(e)}"}), 500


@app.route("/api/temporal_biomarkers", methods=["POST"])
@login_required
def api_temporal_biomarkers():
    """Compute temporal biomarker evolution across the recording and return plots.

    Accepts an EEG CSV file.  Returns grouped time-series plots as base64
    images plus raw numeric biomarker data and summary statistics.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename is None or file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        df = file_service.read_csv(file)
        file_service.validate_eeg_data(df)
        result = temporal_biomarker_service.generate_temporal_plots(df)

        return jsonify({
            "plots": result["plots"],
            "summary": result["summary"],
        }), 200
    except ValueError as e:
        logger.error(f"Validation error in temporal biomarkers: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error computing temporal biomarkers: {e}", exc_info=True)
        return jsonify({"error": f"Failed to compute temporal biomarkers: {str(e)}"}), 500


@app.route("/api/temporal_biomarkers/<int:result_id>", methods=["GET"])
@login_required
def api_temporal_biomarkers_by_result(result_id):
    """Retrieve stored temporal biomarker plots and summary for a result."""
    try:
        plots = temporal_plots_repo.get_by_result(result_id)
        summaries = temporal_summaries_repo.get_by_result(result_id)

        if not plots and not summaries:
            return jsonify({"error": "No temporal biomarker data found for this result"}), 404

        # Reconstruct the response format
        plots_list = [{"group": p["group_name"], "image": p["plot_image"]} for p in plots]
        summary_dict = {}
        for s in summaries:
            summary_dict[s["biomarker_key"]] = {
                "mean": s["mean_value"],
                "std": s["std_value"],
                "min": s["min_value"],
                "max": s["max_value"],
            }

        return jsonify({
            "plots": plots_list,
            "summary": summary_dict,
        }), 200
    except Exception as e:
        logger.error(f"Error fetching temporal biomarkers for result {result_id}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to fetch temporal biomarkers: {str(e)}"}), 500


@app.route("/api/model/info", methods=["GET"])
@login_required
def api_model_info():
    """Return model architecture and hyperparameter information."""
    import json

    try:
        # Load best parameters from JSON file
        params_path = Path(__file__).parent.parent / "best_parameters.json"
        params = {}
        if params_path.exists():
            with open(params_path, "r") as f:
                params = json.load(f)

        # Get model stats if model is loaded
        total_params = 0
        trainable_params = 0
        if hasattr(app, "model_initialized") and app.model_initialized:
            try:
                model = model_loader.model
                if model is not None:
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )
            except Exception as e:
                logger.warning(f"Could not get model stats: {e}")

        return jsonify(
            {
                "params": params,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_loaded": getattr(app, "model_initialized", False),
            }
        ), 200

    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        return jsonify({"error": "Failed to get model info"}), 500


if __name__ == "__main__":
    import os

    # Run Flask app
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t"),
        threaded=True,
    )
