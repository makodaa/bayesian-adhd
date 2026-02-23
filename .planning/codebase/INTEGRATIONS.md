# External Integrations

**Analysis Date:** 2026-02-23

## APIs & External Services

**Frontend CDN Assets:**
- jsDelivr (Bootstrap CDN) - Serves Bootstrap CSS/JS for browser UI pages
  - SDK/Client: Browser `<link>` and `<script>` includes in `backend/app/templates/index.html`, `backend/app/templates/subjects.html`, `backend/app/templates/results.html`, and `backend/app/templates/login.html`
  - Auth: Not applicable

**Third-party API calls from backend:**
- Not detected in Python backend (`requests`, `httpx`, `boto3`, Stripe, Supabase, OpenAI not imported in `backend/app/**/*.py`)

## Data Storage

**Databases:**
- PostgreSQL (Docker image `postgres:18`)
  - Connection: `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_NAME`, `DATABASE_USER`, `DATABASE_PASSWORD`
  - Client: `psycopg2-binary` in `backend/app/db/connection.py`
- Schema managed via SQL file mounted at container init from `backend/app/db/schema.sql` through `docker-compose.yml`

**File Storage:**
- Local filesystem only
  - Persistent Docker volume `backend_exports` mounted to `/app/exports` in `docker-compose.yml` and `docker-compose.prod.yml`
  - Local logs written to `backend/logs/` by `backend/app/core/logging_config.py`

**Caching:**
- None detected (no Redis/Memcached integration)

## Authentication & Identity

**Auth Provider:**
- Custom clinician authentication
  - Implementation: Username/password validation with Werkzeug password hashing in `backend/app/db/repositories/clinicians.py`, session-backed login state in `backend/app/main.py`, and login orchestration in `backend/app/services/clinician_auth_service.py`

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry/Bugsnag/Rollbar integration)

**Logs:**
- Python logging with stdout and rotating file handlers configured in `backend/app/core/logging_config.py`
- Log files: `backend/logs/app.log`, `backend/logs/database.log`, and `backend/logs/ml.log`

## CI/CD & Deployment

**Hosting:**
- Container-based self-hosting target via Docker Compose (`docker-compose.prod.yml`)

**CI Pipeline:**
- Not detected (`.github/workflows/` not present)

## Environment Configuration

**Required env vars:**
- Database: `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_NAME`, `DATABASE_USER`, `DATABASE_PASSWORD` (read in `backend/app/db/connection.py`)
- App runtime: `PORT`, `FLASK_DEBUG` (read in `backend/app/main.py`)
- Compose-level DB secret mapping: `DB_PASSWORD` used in `docker-compose.prod.yml` for service env interpolation

**Secrets location:**
- `.env` file present at `backend/.env` (environment configuration file present; contents intentionally not inspected)
- Runtime injection also configured via Compose environment blocks in `docker-compose.yml` and `docker-compose.prod.yml`

## Webhooks & Callbacks

**Incoming:**
- None detected for webhook-style callback endpoints
- Standard application HTTP endpoints are exposed from Flask in `backend/app/main.py` (for example `/predict`, `/api/results`, `/api/topographic_maps`, `/api/temporal_biomarkers`)

**Outgoing:**
- None detected (no outbound webhook dispatch or third-party callback clients found)

---

*Integration audit: 2026-02-23*
