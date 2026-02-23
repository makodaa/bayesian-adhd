# Coding Conventions

**Analysis Date:** 2026-02-23

## Naming Patterns

**Files:**
- Use `snake_case.py` for Python modules in `backend/app/services/` (for example `backend/app/services/recording_service.py`) and `backend/app/db/repositories/` (for example `backend/app/db/repositories/temporal_summaries.py`).
- Use lowercase HTML filenames for routed pages in `backend/app/templates/` (for example `backend/app/templates/login.html`).

**Functions:**
- Use `snake_case` for Python functions and methods in `backend/app/main.py`, `backend/app/services/eeg_service.py`, and `backend/app/db/repositories/clinicians.py`.
- Use `camelCase` for inline browser JavaScript functions in `backend/app/templates/index.html` (for example `loadTopographicMaps`, `displayBandPowers`).

**Variables:**
- Use `snake_case` for Python local variables and parameters in `backend/app/services/file_service.py` and `backend/app/db/connection.py`.
- Use `UPPER_SNAKE_CASE` for module constants in `backend/app/config.py` and `backend/app/core/logging_config.py`.

**Types:**
- Use `PascalCase` for classes in `backend/app/services/subject_service.py`, `backend/app/ml/model_loader.py`, and `backend/app/db/repositories/base.py`.
- Use occasional modern Python union typing (`X | None`) in `backend/app/main.py` and `backend/app/ml/model_loader.py`; mixed with `Optional[...]` in `backend/app/services/results_service.py`.

## Code Style

**Formatting:**
- Not detected: no `pyproject.toml`, `setup.cfg`, `.flake8`, `.pylintrc`, `mypy.ini`, or formatter config in repository root.
- Follow existing style in `backend/app/main.py` and `backend/app/services/eeg_service.py`: 4-space indentation, triple-quoted docstrings for public functions, and grouped section comments.

**Linting:**
- No Python linter configuration detected in repository root or `backend/`.
- HTML linting hints are configured via `.hintrc` and `backend/.hintrc`; use those rules when editing templates in `backend/app/templates/index.html` and `backend/app/templates/login.html`.

## Import Organization

**Order:**
1. Python standard library imports first (for example `pathlib`, `typing`) in `backend/app/main.py` and `backend/app/ml/model_loader.py`.
2. Third-party imports second (for example `flask`, `numpy`, `torch`) in `backend/app/main.py` and `backend/app/services/eeg_service.py`.
3. Local package imports last (for example `from .services...`, `from ..db...`) in `backend/app/main.py` and `backend/app/services/recording_service.py`.

**Path Aliases:**
- Not used; imports are package-relative (for example `from ..core.logging_config import get_app_logger` in `backend/app/services/file_service.py`).

## Error Handling

**Patterns:**
- Use explicit `try/except` blocks in route handlers and repository operations, returning HTTP JSON errors in `backend/app/main.py` and re-raising in `backend/app/db/repositories/results.py`.
- Use typed validation failures (`ValueError`) for bad inputs in `backend/app/services/file_service.py` and `backend/app/services/eeg_service.py`.
- `backend/app/core/error_handle.py` provides a decorator that catches broad exceptions and returns `{"error": ...}` payloads for helper functions in `backend/app/main.py`.

## Logging

**Framework:** Python `logging` with rotating file handlers in `backend/app/core/logging_config.py`.

**Patterns:**
- Initialize per-module loggers via `get_app_logger`, `get_db_logger`, or `get_ml_logger` in `backend/app/main.py`, `backend/app/db/connection.py`, and `backend/app/ml/model_loader.py`.
- Log lifecycle events with `info`, detail traces with `debug`, and include `exc_info=True` for failures in `backend/app/services/clinician_auth_service.py` and `backend/app/db/repositories/clinicians.py`.
- Avoid introducing new `print(...)` debugging in application paths; existing mixed usage appears in `backend/app/main.py`, `backend/app/services/eeg_service.py`, and `backend/app/utils/timer.py`.

## Comments

**When to Comment:**
- Use comments to separate processing phases in long functions (for example in `backend/app/main.py:467`, `backend/app/services/eeg_service.py:214`, and `backend/app/templates/index.html:1638`).
- Keep comments practical and operation-focused (for example SQL intent in `backend/app/db/repositories/subjects.py` and pipeline steps in `backend/scripts/validation_data_script.py`).

**JSDoc/TSDoc:**
- Not applicable for TypeScript; JavaScript in `backend/app/templates/index.html` uses line comments rather than JSDoc blocks.
- Python uses docstrings for many classes/functions (for example `backend/app/services/recording_service.py`, `backend/app/db/connection.py`, and `backend/app/ml/model.py`).

## Function Design

**Size:**
- Keep service/repository methods focused and short (see `backend/app/services/file_service.py` and `backend/app/db/repositories/base.py`).
- Route handlers in `backend/app/main.py` can be large; prefer extracting helper functions for new logic instead of extending `predict()` further.

**Parameters:**
- Use typed parameters in service-layer methods where possible (examples in `backend/app/services/results_service.py` and `backend/app/services/subject_service.py`).
- Accept flexible payload dictionaries for request-driven data in `backend/app/services/clinician_service.py`.

**Return Values:**
- Return primitive IDs from create operations in repositories and services (examples in `backend/app/db/repositories/results.py` and `backend/app/services/recording_service.py`).
- Return dictionary payloads for API transport and computed outputs in `backend/app/main.py` and `backend/app/services/eeg_service.py`.

## Module Design

**Exports:**
- Modules export classes/functions directly; no explicit export lists detected (examples: `backend/app/services/subject_service.py`, `backend/app/ml/model_loader.py`).

**Barrel Files:**
- Minimal `__init__.py` files exist in `backend/app/`, `backend/app/services/`, and `backend/app/db/repositories/`; they are package markers and not aggregation barrels.

---

*Convention analysis: 2026-02-23*
