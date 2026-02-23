# Codebase Concerns

**Analysis Date:** 2026-02-23

## Tech Debt

**Monolithic HTTP and orchestration layer:**
- Issue: Route handling, auth/session checks, file processing, ML orchestration, DB writes, and response shaping are concentrated in one module.
- Files: `backend/app/main.py`
- Impact: High change risk, difficult debugging, and frequent merge conflicts around one hotspot file.
- Fix approach: Split `backend/app/main.py` into route blueprints and move orchestration into focused service modules with narrower responsibilities.

**Duplicated EEG computation logic across services/scripts:**
- Issue: Similar band-power and signal processing logic is implemented in multiple places.
- Files: `backend/app/services/eeg_service.py`, `backend/app/services/band_analysis_service.py`, `backend/app/services/topographic_service.py`, `backend/scripts/validation_data_script.py`
- Impact: Behavior drift and inconsistent metrics when one path changes without updating others.
- Fix approach: Centralize shared spectral computations into one reusable module and consume it from services/scripts.

**Stubbed reporting service remains in production tree:**
- Issue: Public methods raise `NotImplementedError` and include commented-out pseudo-implementation.
- Files: `backend/app/services/report_service.py`
- Impact: Any wiring to this service fails at runtime and increases uncertainty about supported features.
- Fix approach: Either implement end-to-end behavior with tests or remove the service until it is fully supported.

**Dependency hygiene is weak:**
- Issue: Dependencies are unpinned and include duplicates.
- Files: `backend/requirements.txt`
- Impact: Non-reproducible builds and accidental breaking upgrades across environments.
- Fix approach: Pin versions, remove duplicates (`scikit-learn` appears twice), and maintain a lock strategy.

**Database bootstrap path is ambiguous:**
- Issue: The root DB init file is empty while the active schema exists elsewhere.
- Files: `database/init.sql`, `backend/app/db/schema.sql`
- Impact: Fresh environments can start without expected tables if bootstrap wiring points to the empty file.
- Fix approach: Make one canonical schema/init path and enforce it in deployment/bootstrap scripts.

## Known Bugs

**About page calls a missing API endpoint:**
- Symptoms: About page fails to load clinician display info and logs a client-side fallback message.
- Files: `backend/app/templates/about.html`, `backend/app/main.py`
- Trigger: Opening `about.html` triggers `fetch("/api/clinician/current")`, but no matching route exists.
- Workaround: Use `/api/me` (already implemented in `backend/app/main.py`) and map response fields accordingly.

**Login endpoint returns 500 on invalid JSON payload shape:**
- Symptoms: Server error instead of validation error when request body is absent/non-JSON.
- Files: `backend/app/main.py`
- Trigger: `request.get_json()` can return `None`, then `data.get(...)` raises an exception.
- Workaround: Require JSON and guard `if not isinstance(data, dict)` before field access.

**Results repository query references a non-existent timestamp column:**
- Symptoms: Query failure if `get_by_recording()` is called.
- Files: `backend/app/db/repositories/results.py`, `backend/app/db/schema.sql`
- Trigger: `ORDER BY created_at DESC` is used, but `results` table defines `inferenced_at`.
- Workaround: Replace `created_at` with `inferenced_at` in `backend/app/db/repositories/results.py`.

## Security Considerations

**Hardcoded secrets and insecure defaults in codebase:**
- Risk: Predictable session signing and fallback DB credentials increase compromise risk if env config is missing.
- Files: `backend/app/main.py`, `backend/app/db/connection.py`, `backend/.env` (present)
- Current mitigation: Environment variables are supported for DB config in `backend/app/db/connection.py`.
- Recommendations: Remove hardcoded `app.secret_key`, fail fast when required env vars are absent, and ensure secrets live outside source control.

**Authorization model is weak for admin operations:**
- Risk: Admin access is determined by display name string equality (`"Admin Clinician"`) instead of role claims.
- Files: `backend/app/main.py`, `backend/app/db/schema.sql`
- Current mitigation: Route is login-protected via `@login_required`.
- Recommendations: Add explicit role column/claims, enforce role-based authorization server-side, and remove name-based privilege checks.

**Session-auth API surface lacks CSRF/rate-limiting controls:**
- Risk: Session-backed POST routes can be targeted by CSRF; login endpoint allows brute-force attempts.
- Files: `backend/app/main.py`
- Current mitigation: Password hashes are verified with Werkzeug in `backend/app/db/repositories/clinicians.py`.
- Recommendations: Add CSRF protection for state-changing routes and request throttling/lockout on `/api/login`.

**Sensitive data endpoints are exposed without auth guards:**
- Risk: Subject/clinician/result metadata can be queried without `@login_required`.
- Files: `backend/app/main.py`
- Current mitigation: UI pages are login-gated, but several JSON endpoints are not.
- Recommendations: Require authentication on all patient/result management APIs and return least-privilege fields.

**Default administrator credential is documented in schema comment:**
- Risk: Predictable bootstrap credentials are discoverable and likely reused in non-prod/prod drift scenarios.
- Files: `backend/app/db/schema.sql`
- Current mitigation: Stored value is hashed, not plaintext.
- Recommendations: Remove fixed default admin bootstrap, require one-time credential provisioning at deployment.

## Performance Bottlenecks

**Predict request does heavy synchronous CPU work in request thread:**
- Problem: Filtering, ICA, model inference, band analysis, topomap generation, temporal plots, and DB writes run in one HTTP cycle.
- Files: `backend/app/main.py`, `backend/app/services/eeg_service.py`, `backend/app/services/topographic_service.py`, `backend/app/services/temporal_biomarker_service.py`
- Cause: No background queue/offloading; computation and persistence are tightly coupled to request latency.
- Improvement path: Move heavy analysis to background jobs (Celery/RQ), return job IDs, and stream/poll status.

**N+1 query pattern in subject listing endpoint:**
- Problem: Each subject triggers an additional recordings query.
- Files: `backend/app/main.py`, `backend/app/services/recording_service.py`, `backend/app/db/repositories/recordings.py`
- Cause: `get_subjects()` loops through subjects and calls `get_recordings_by_subject()` per row.
- Improvement path: Fetch recording counts in one aggregate SQL query and return a joined dataset.

**Large base64 images are persisted in TEXT columns:**
- Problem: Storage and response payload sizes grow quickly for temporal/topographic artifacts.
- Files: `backend/app/db/schema.sql`, `backend/app/db/repositories/topographic_maps.py`, `backend/app/db/repositories/temporal_plots.py`, `backend/app/services/topographic_service.py`, `backend/app/services/temporal_biomarker_service.py`
- Cause: Binary image payloads are encoded and stored inline in relational rows.
- Improvement path: Store images in object storage/filesystem and keep only references/metadata in DB.

**Visualization path uses high-DPI figure generation:**
- Problem: Rendering at 600 DPI increases CPU time and memory per request.
- Files: `backend/app/services/visualization_service.py`
- Cause: `fig.savefig(..., dpi=600)` for potentially long multichannel recordings.
- Improvement path: Reduce DPI, cap output dimensions, and add request-level limits/caching.

## Fragile Areas

**Broad exception handling hides root causes and mixes response behavior:**
- Files: `backend/app/main.py`, `backend/app/core/error_handle.py`, `backend/app/services/clinician_auth_service.py`
- Why fragile: Many `except Exception` blocks convert distinct failure modes into generic messages, making diagnosis and client handling inconsistent.
- Safe modification: Replace broad catches with typed exceptions and centralized error-to-HTTP mapping.
- Test coverage: No automated tests detected for error-path behavior.

**Partial-success writes around advanced artifacts:**
- Files: `backend/app/main.py`
- Why fragile: Topographic/temporal generation failures are swallowed with warnings while core prediction succeeds, creating inconsistent result completeness.
- Safe modification: Track artifact generation status explicitly in DB/API and expose partial state to clients.
- Test coverage: No automated contract tests detected for partial-result scenarios.

**Channel assumptions enforced with assertions and strict shape logic:**
- Files: `backend/app/services/eeg_service.py`, `backend/app/config.py`
- Why fragile: Processing assumes exact channel set/shape; atypical data can fail abruptly or produce hard-to-debug behavior.
- Safe modification: Validate schema up front with explicit user-facing errors and fallback mapping policies.
- Test coverage: No fixture-based compatibility tests detected for alternate channel naming/layouts.

## Scaling Limits

**Worker-level throughput ceiling for CPU-bound analysis:**
- Current capacity: One analysis request can occupy a worker for the full processing lifecycle; production command config uses 4 Gunicorn workers.
- Limit: Concurrent analysis requests quickly saturate workers and increase queueing latency.
- Scaling path: Offload CPU-heavy analysis to dedicated worker processes and keep web workers for lightweight API handling.

**Unbounded upload and in-memory processing path:**
- Current capacity: Upload is read fully into memory before parsing.
- Limit: Large/abusive files can pressure memory and degrade service for all users.
- Scaling path: Enforce upload size limits, stream/ chunk parse where possible, and reject oversized files early.

## Dependencies at Risk

**Unpinned scientific/ML stack:**
- Risk: Upstream version changes can alter numerical behavior, model loading behavior, or break APIs.
- Impact: Reproducibility and deployment stability degrade over time.
- Migration plan: Pin and periodically refresh versions in `backend/requirements.txt`; validate via smoke/integration tests before upgrades.

**`psycopg2-binary` used as runtime dependency:**
- Risk: Binary wheel package is convenient but discouraged by upstream for long-term production packaging.
- Impact: Potential compatibility/operational issues across target environments.
- Migration plan: Move to source build `psycopg2` with explicit system dependencies in image build.

## Missing Critical Features

**Automated test suite and CI quality gates are not present:**
- Problem: There is no detected test directory or test files for backend services/routes.
- Blocks: Safe refactoring, regression detection, and confident releases.

**Robust RBAC and audit trail capabilities are missing:**
- Problem: Admin privilege is name-based and not role/permission-based.
- Blocks: Secure multi-user operation and compliance-oriented access controls.

**Asynchronous job processing for long-running analyses is missing:**
- Problem: Heavy analysis endpoints are synchronous.
- Blocks: Reliable performance under concurrent clinical workloads.

## Test Coverage Gaps

**API route behavior is untested:**
- What's not tested: Authentication flows, payload validation, authorization checks, and error responses.
- Files: `backend/app/main.py`
- Risk: Security regressions and API breakages can ship unnoticed.
- Priority: High

**Signal processing and model inference pipeline is untested:**
- What's not tested: Filtering/ICA behavior, feature extraction consistency, and model input-shape edge cases.
- Files: `backend/app/services/eeg_service.py`, `backend/app/services/band_analysis_service.py`, `backend/app/services/topographic_service.py`, `backend/app/services/temporal_biomarker_service.py`
- Risk: Silent scientific drift or runtime failures from data-shape variation.
- Priority: High

**Repository SQL contract integrity is untested:**
- What's not tested: Query correctness against schema (including timestamp column names and joins).
- Files: `backend/app/db/repositories/results.py`, `backend/app/db/repositories/recordings.py`, `backend/app/db/repositories/subjects.py`, `backend/app/db/schema.sql`
- Risk: Latent query bugs surface only in production traffic.
- Priority: Medium

---

*Concerns audit: 2026-02-23*
