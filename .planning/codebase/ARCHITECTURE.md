# Architecture

**Analysis Date:** 2026-02-23

## Pattern Overview

**Overall:** Layered monolith (Flask MVC-style web app + service/repository backend)

**Key Characteristics:**
- Route handlers and orchestration are centralized in `backend/app/main.py`.
- Business logic is split into focused service classes under `backend/app/services/`.
- Persistence uses repository classes over raw SQL and psycopg2 in `backend/app/db/repositories/` and `backend/app/db/connection.py`.

## Layers

**Presentation + HTTP Routing Layer:**
- Purpose: Define web pages and JSON API endpoints, parse requests, enforce auth, and shape responses.
- Location: `backend/app/main.py`, templates in `backend/app/templates/`.
- Contains: Flask app construction (`Application`), route decorators, session-based auth (`login_required`), request validation.
- Depends on: Service layer (`backend/app/services/*.py`), repositories for direct retrieval in a few endpoints, Flask session/request/response APIs.
- Used by: Browser clients and API consumers via routes such as `/predict`, `/api/results`, `/api/topographic_maps`.

**Service (Business Logic) Layer:**
- Purpose: Encapsulate domain operations (EEG classification, band analysis, clinician auth, report generation).
- Location: `backend/app/services/`.
- Contains: `EEGService`, `BandAnalysisService`, `TopographicService`, `TemporalBiomarkerService`, `FileService`, `ResultsService`, `PDFReportService`.
- Depends on: Repositories, ML loader/model, utility functions, scientific stack (NumPy/SciPy/MNE/Torch/Matplotlib).
- Used by: `backend/app/main.py` route handlers.

**Data Access Layer:**
- Purpose: Own SQL read/write operations and DB transaction boundaries.
- Location: `backend/app/db/repositories/`, shared base in `backend/app/db/repositories/base.py`, connection manager in `backend/app/db/connection.py`.
- Contains: table-specific repositories (`subjects.py`, `recordings.py`, `results.py`, `band_powers.py`, `ratios.py`, `topographic_maps.py`, `temporal_plots.py`, `temporal_summaries.py`).
- Depends on: psycopg2 connection context manager (`get_db_connection`) and schema in `backend/app/db/schema.sql`.
- Used by: Service layer and some route handlers in `backend/app/main.py`.

**ML + Signal Processing Layer:**
- Purpose: Load trained model, transform EEG windows, run inference, compute biomarkers/plots.
- Location: `backend/app/ml/`, `backend/app/services/eeg_service.py`, `backend/app/services/topographic_service.py`, `backend/app/services/temporal_biomarker_service.py`, `backend/app/utils/signal_processing.py`.
- Contains: `EEGCNNLSTM` model definition (`backend/app/ml/model.py`), model loading (`backend/app/ml/model_loader.py`), feature extraction and filtering.
- Depends on: Config constants in `backend/app/config.py`, model weights `backend/app/ml/optimized_model.pth`.
- Used by: `EEGService` in prediction flow and visualization/biomarker endpoints.

**Cross-Cutting Infrastructure Layer:**
- Purpose: Logging, error wrapping, threading helper, runtime/container entrypoints.
- Location: `backend/app/core/`, runtime configs in `backend/Dockerfile`, `docker-compose.yml`, `docker-compose.prod.yml`.
- Contains: logger factory (`backend/app/core/logging_config.py`), decorator error wrapper (`backend/app/core/error_handle.py`), thread executor decorator (`backend/app/core/threader.py`).
- Depends on: stdlib logging/concurrency and file system (`backend/logs/`).
- Used by: All layers via imported logger/getters and decorators.

## Data Flow

**Prediction + Persistence Flow (`POST /predict`):**

1. `backend/app/main.py` validates multipart request fields and EEG file, then parses CSV via `FileService.read_csv` in `backend/app/services/file_service.py`.
2. Subject and recording are created/resolved through `SubjectService` (`backend/app/services/subject_service.py`) and `RecordingService` (`backend/app/services/recording_service.py`), which call repositories in `backend/app/db/repositories/subjects.py` and `backend/app/db/repositories/recordings.py`.
3. `EEGService.classify_and_save` (`backend/app/services/eeg_service.py`) applies filtering/ICA/feature extraction, runs `ModelLoader.model` from `backend/app/ml/model_loader.py`, stores prediction with `ResultsRepository` in `backend/app/db/repositories/results.py`.
4. Additional analyses run in the same request: band powers (`BandAnalysisService`), topographic maps (`TopographicService`), temporal biomarkers (`TemporalBiomarkerService`) and are persisted through corresponding repositories.
5. Route returns normalized JSON payload containing classification, confidence, IDs, and band analysis for UI rendering in `backend/app/templates/index.html`.

**Visualization Flow (`POST /visualize_all_eeg` and `/visualize_eeg/<type>`):**

1. Client-side JS in `backend/app/templates/index.html` uploads file.
2. `backend/app/main.py` validates extension and data.
3. `VisualizationService.visualize_df` in `backend/app/services/visualization_service.py` filters channels and renders MNE plots to base64.
4. Route responds with image arrays consumed directly by template JS.

**Results Retrieval Flow (`GET /api/results/<id>`):**

1. `backend/app/main.py` calls `ResultsService.get_result_with_full_details` in `backend/app/services/results_service.py`.
2. Service executes joined SQL plus follow-up queries for `band_powers` and `ratios` using DB connection from `backend/app/db/connection.py`.
3. UI in `backend/app/templates/results.html` fetches and renders the modal view; optional PDF is generated via `/api/results/<id>/pdf` using `PDFReportService`.

**State Management:**
- Server-side auth/session state uses Flask session cookies with filesystem-backed session config in `backend/app/main.py`.
- Frontend page state is local and script-managed in template JS (`analysisData`, modal state, selected result) in `backend/app/templates/index.html` and `backend/app/templates/results.html`.
- Persistence state is PostgreSQL tables defined in `backend/app/db/schema.sql`.

## Key Abstractions

**Repository Abstraction:**
- Purpose: Isolate table-level SQL from services/routes.
- Examples: `backend/app/db/repositories/results.py`, `backend/app/db/repositories/subjects.py`, `backend/app/db/repositories/topographic_maps.py`.
- Pattern: Inherit `BaseRepository` (`backend/app/db/repositories/base.py`) and use `with self.get_connection()` transaction scope.

**Service Abstraction:**
- Purpose: Keep route handlers thin; compose multi-step workflows.
- Examples: `backend/app/services/eeg_service.py`, `backend/app/services/band_analysis_service.py`, `backend/app/services/clinician_auth_service.py`.
- Pattern: Constructor injection of repositories and collaborators, with each service centered on a bounded capability.

**ModelLoader Abstraction:**
- Purpose: Own model instantiation + weight loading + parameter defaults.
- Examples: `backend/app/ml/model_loader.py`, model class in `backend/app/ml/model.py`.
- Pattern: Singleton-like app-level instance created in `backend/app/main.py` and initialized lazily in `@app.before_request`.

**Template-as-Frontend Abstraction:**
- Purpose: Deliver server-rendered HTML pages with rich inline JavaScript behavior.
- Examples: `backend/app/templates/index.html`, `backend/app/templates/results.html`, `backend/app/templates/login.html`.
- Pattern: No separate frontend build pipeline; each template embeds CSS + JS and calls JSON endpoints directly.

## Entry Points

**Application Runtime Entry Point:**
- Location: `backend/app/main.py`.
- Triggers: Flask CLI in development (`backend/Dockerfile` development CMD) and Gunicorn in production (`backend/Dockerfile` production CMD).
- Responsibilities: App construction, singleton dependency wiring, route definitions, model initialization hook.

**Container Entry Points:**
- Location: `docker-compose.yml`, `docker-compose.prod.yml`, `backend/Dockerfile`.
- Triggers: `docker compose up` (dev/prod).
- Responsibilities: Start PostgreSQL and backend service, mount schema init script (`backend/app/db/schema.sql`), expose `8000` and `5432`.

**Database Bootstrap Entry Point:**
- Location: `backend/app/db/schema.sql` mounted as init SQL from compose files.
- Triggers: First PostgreSQL container initialization.
- Responsibilities: Create core tables (`subjects`, `recordings`, `results`, `band_powers`, `ratios`, `topographic_maps`, `temporal_plots`, `temporal_summaries`, `clinicians`, `clinician_sessions`, `reports`).

## Error Handling

**Strategy:** Mixed per-route try/except + decorator wrapping + DB transaction rollback.

**Patterns:**
- Route-level explicit exception mapping to HTTP status codes in `backend/app/main.py`.
- Decorator-based generic error dict return in `backend/app/core/error_handle.py` for helper wrappers.
- DB context manager rollback/rethrow in `backend/app/db/connection.py`.

## Cross-Cutting Concerns

**Logging:** Structured module loggers via `get_app_logger` / `get_db_logger` / `get_ml_logger` in `backend/app/core/logging_config.py`, with rotating files under `backend/logs/`.
**Validation:** Request field checks in `backend/app/main.py` plus file/data checks in `backend/app/services/file_service.py` and channel checks in `backend/app/services/eeg_service.py`.
**Authentication:** Session-based clinician login/logout through `/api/login` and `/api/logout` in `backend/app/main.py`, credential verification in `backend/app/services/clinician_auth_service.py`, active-session tracking in `backend/app/db/repositories/clinicians.py`.

---

*Architecture analysis: 2026-02-23*
