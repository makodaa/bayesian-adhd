# Technology Stack

**Analysis Date:** 2026-02-23

## Languages

**Primary:**
- Python 3.11 - Backend application, ML inference, data processing, and PDF generation in `backend/app/main.py`, `backend/app/services/eeg_service.py`, and `backend/app/services/pdf_service.py`

**Secondary:**
- SQL (PostgreSQL dialect) - Database schema and relational modeling in `backend/app/db/schema.sql`
- HTML/CSS/JavaScript - Server-rendered UI templates and client-side API calls in `backend/app/templates/index.html`, `backend/app/templates/results.html`, and `backend/app/static/styles.css`

## Runtime

**Environment:**
- Python runtime is pinned by container base image `python:3.11-slim` in `backend/Dockerfile`
- Web server runtime is Flask in development and Gunicorn in production via `backend/Dockerfile`

**Package Manager:**
- pip (installed/used in container build) in `backend/Dockerfile`
- Lockfile: missing (no `poetry.lock`, `Pipfile.lock`, or pinned lockfile detected)

## Frameworks

**Core:**
- Flask (version not pinned in `requirements.txt`) - HTTP routes, template rendering, and JSON API in `backend/app/main.py`
- Bootstrap 5.3.x via CDN - UI framework loaded in templates such as `backend/app/templates/index.html` and `backend/app/templates/subjects.html`

**Testing:**
- Not detected (no pytest/unittest config or test runner config files found)

**Build/Dev:**
- Docker multi-stage build for development/production targets in `backend/Dockerfile`
- Docker Compose orchestration for local and production-like environments in `docker-compose.yml` and `docker-compose.prod.yml`
- gunicorn (from `backend/requirements.txt`) - Production WSGI server started in `backend/Dockerfile`

## Key Dependencies

**Critical:**
- `torch` (version not pinned) - Model architecture and inference in `backend/app/ml/model.py` and `backend/app/ml/model_loader.py`
- `mne` (version not pinned) - EEG-oriented processing and visualization logic in `backend/app/services/eeg_service.py` and `backend/app/services/topographic_service.py`
- `numpy`, `scipy`, `pandas` (versions not pinned) - Numeric processing, signal transforms, and CSV data handling in `backend/app/config.py`, `backend/app/services/band_analysis_service.py`, and `backend/app/services/file_service.py`
- `flask` (version not pinned) - Request handling and API surface in `backend/app/main.py`

**Infrastructure:**
- `psycopg2-binary` (version not pinned) - PostgreSQL connectivity in `backend/app/db/connection.py`
- `werkzeug` (version not pinned) - Password hashing and request-related utilities in `backend/app/db/repositories/clinicians.py`
- `reportlab` (version not pinned) - PDF report rendering in `backend/app/services/pdf_service.py`
- `matplotlib` (version not pinned) - Plot generation using non-GUI backend in `backend/app/main.py` and plotting services

## Configuration

**Environment:**
- Application reads runtime values from environment variables: `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_NAME`, `DATABASE_USER`, `DATABASE_PASSWORD`, `PORT`, and `FLASK_DEBUG` in `backend/app/db/connection.py` and `backend/app/main.py`
- `.env` file present at `backend/.env` (environment configuration file present; contents intentionally not inspected)
- Compose files define additional environment defaults and container env wiring in `docker-compose.yml` and `docker-compose.prod.yml`

**Build:**
- Container build and start commands in `backend/Dockerfile`
- Service composition and volume/network/runtime config in `docker-compose.yml` and `docker-compose.prod.yml`
- HTMLHint config for frontend linting in `.hintrc` and `backend/.hintrc`

## Platform Requirements

**Development:**
- Docker and Docker Compose workflow documented in `README.md` and implemented via `docker-compose.yml`
- Host capable of running Python scientific stack dependencies (`gcc`, `build-essential`) included in `backend/Dockerfile`

**Production:**
- Containerized deployment target using `docker-compose.prod.yml` with `backend` and `database` services
- Gunicorn process serving Flask app in production container target in `backend/Dockerfile`

---

*Stack analysis: 2026-02-23*
