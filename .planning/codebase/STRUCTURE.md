# Codebase Structure

**Analysis Date:** 2026-02-23

## Directory Layout

```text
bayesian-adhd/
├── backend/                 # Flask app, services, DB access, ML logic
│   ├── app/                 # Runtime application package
│   │   ├── core/            # Logging, error wrappers, threading helper
│   │   ├── db/              # DB connection, SQL schema, repositories
│   │   ├── ml/              # Model definition, loader, model weights
│   │   ├── services/        # Domain/business logic services
│   │   ├── static/          # Shared static assets (CSS/images)
│   │   ├── templates/       # Server-rendered HTML + inline JS/CSS
│   │   ├── utils/           # Signal-processing and helper utilities
│   │   ├── config.py        # App constants (sampling, bands, paths)
│   │   └── main.py          # Main Flask app and route entrypoint
│   ├── exports/             # Runtime export artifacts
│   ├── notebooks/           # Training/experiment notebooks
│   ├── scripts/             # One-off utility scripts
│   ├── logs/                # Rotating runtime logs
│   ├── requirements.txt     # Python dependency list
│   └── Dockerfile           # Dev/prod container build
├── database/                # Extra DB assets (init.sql currently empty)
├── devlogs/                 # Project notes/log files
├── playground/              # Planning artifacts
├── docker-compose.yml       # Development orchestration
├── docker-compose.prod.yml  # Production orchestration
└── README.md                # Setup and usage overview
```

## Directory Purposes

**`backend/app/`:**
- Purpose: Main application package imported as `app` by Flask/Gunicorn.
- Contains: HTTP routes (`backend/app/main.py`), services (`backend/app/services/`), repositories (`backend/app/db/repositories/`), model code (`backend/app/ml/`), templates (`backend/app/templates/`).
- Key files: `backend/app/main.py`, `backend/app/config.py`, `backend/app/db/connection.py`.

**`backend/app/services/`:**
- Purpose: Implement business operations and computational workflows.
- Contains: `*_service.py` modules such as `backend/app/services/eeg_service.py`, `backend/app/services/band_analysis_service.py`, `backend/app/services/pdf_service.py`.
- Key files: `backend/app/services/eeg_service.py`, `backend/app/services/results_service.py`, `backend/app/services/topographic_service.py`.

**`backend/app/db/repositories/`:**
- Purpose: Table-specific SQL access classes.
- Contains: One repository per table/domain (`subjects.py`, `recordings.py`, `results.py`, `ratios.py`, `topographic_maps.py`, etc.) plus `base.py`.
- Key files: `backend/app/db/repositories/base.py`, `backend/app/db/repositories/results.py`, `backend/app/db/repositories/clinicians.py`.

**`backend/app/templates/`:**
- Purpose: End-user pages and client-side behavior.
- Contains: `.html` pages with embedded CSS/JS for analysis, login, and management UIs.
- Key files: `backend/app/templates/index.html`, `backend/app/templates/results.html`, `backend/app/templates/login.html`.

**`backend/app/ml/`:**
- Purpose: ML model architecture and runtime loading.
- Contains: `backend/app/ml/model.py`, `backend/app/ml/model_loader.py`, `backend/app/ml/optimized_model.pth`.
- Key files: `backend/app/ml/model_loader.py`, `backend/app/ml/model.py`.

**`backend/app/core/`:**
- Purpose: Shared operational utilities used across modules.
- Contains: logger setup, decorator-based error handling, and thread helper.
- Key files: `backend/app/core/logging_config.py`, `backend/app/core/error_handle.py`, `backend/app/core/threader.py`.

## Key File Locations

**Entry Points:**
- `backend/app/main.py`: Flask application object, dependency wiring, and all route registrations.
- `backend/Dockerfile`: Container runtime commands for dev (`flask run`) and prod (`gunicorn`).
- `docker-compose.yml`: Starts `backend` and `database` services for development.

**Configuration:**
- `backend/app/config.py`: Signal-processing constants and model path.
- `backend/.env`: Environment overrides for local backend runtime (file present; treat as secret config, do not read into docs).
- `docker-compose.prod.yml`: Production service environment and startup behavior.

**Core Logic:**
- `backend/app/services/eeg_service.py`: Classification pipeline.
- `backend/app/services/band_analysis_service.py`: Band power/ratio computation and persistence.
- `backend/app/services/topographic_service.py`: Scalp topomap generation and reconstruction.
- `backend/app/services/temporal_biomarker_service.py`: Sliding-window biomarker evolution.

**Testing:**
- Not detected: no `tests/` directory and no `*.test.py`/`*.spec.py` files were found in the mapped tree.

## Naming Conventions

**Files:**
- Use snake_case module names for Python code: `backend/app/services/recording_service.py`.
- Use plural nouns for table repositories: `backend/app/db/repositories/subjects.py`, `backend/app/db/repositories/results.py`.
- Use lowercase HTML page names tied to routes: `backend/app/templates/subjects.html`, `backend/app/templates/about.html`.

**Directories:**
- Use lowercase domain folders under `backend/app/`: `services`, `repositories`, `templates`, `utils`.
- Keep layered boundaries explicit by directory (`core` vs `services` vs `db` vs `ml`).

## Where to Add New Code

**New Feature:**
- Primary code: add orchestration endpoint in `backend/app/main.py`, business logic in a new/existing module under `backend/app/services/`, and SQL operations in `backend/app/db/repositories/`.
- Tests: create a new top-level `backend/tests/` directory (not currently present) and mirror module names (`test_<module>.py`).

**New Component/Module:**
- Implementation: place reusable domain behavior in `backend/app/services/<feature>_service.py`.
- Persistence companion: add `backend/app/db/repositories/<feature_plural>.py` and update schema in `backend/app/db/schema.sql` when new tables are required.

**Utilities:**
- Shared helpers: put low-level pure helpers in `backend/app/utils/` (for data/math helpers) or `backend/app/core/` (for runtime/logging/error/threading concerns).

## Special Directories

**`backend/notebooks/`:**
- Purpose: Model experimentation and training artifacts (`*.ipynb`).
- Generated: No (manually maintained notebooks).
- Committed: Yes.

**`backend/logs/`:**
- Purpose: Runtime log output (`app.log`, `database.log`, `ml.log`).
- Generated: Yes.
- Committed: No (`logs` ignored in `.gitignore`).

**`backend/exports/`:**
- Purpose: Exported/generated outputs from processing workflows.
- Generated: Yes.
- Committed: No (`exports` ignored in `.gitignore`).

**`.secrets/`:**
- Purpose: Local secret storage directory (present at repository root).
- Generated: Environment-specific.
- Committed: No (`.secrets` ignored in `.gitignore`).

---

*Structure analysis: 2026-02-23*
