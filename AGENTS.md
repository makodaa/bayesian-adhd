# AGENTS Guide for `bayesian-adhd`
This guide is for coding agents working in this repository.
It documents reliable commands plus style/convention expectations inferred from the codebase.

## 1) Repository Map
- Compose services at repo root: `database`, `backend`.
- Flask backend lives in `backend/app/`.
- Flask app entrypoint: `backend/app/main.py` (`app.main:app`).
- Database schema: `backend/app/db/schema.sql`.
- Repositories: `backend/app/db/repositories/`.
- Business logic services: `backend/app/services/`.
- Frontend templates: `backend/app/templates/`.
- Helper scripts: `backend/scripts/`.

## 2) Environment and Setup
- Prefer repo virtualenv: `.venv/`.
- Docker Python base image: `python:3.11-slim`.
- Install backend dependencies:
```bash
cd backend
../.venv/bin/pip install -r requirements.txt
```
- `pytest` is available in this env, but not pinned in `backend/requirements.txt`.

## 3) Build and Run Commands
### Full stack via Docker (recommended)
```bash
docker compose up --build
docker compose up -d
docker compose logs -f backend
docker compose down
docker compose down -v && docker compose up --build
```

### Backend-only local run
```bash
../.venv/bin/python -m flask --app app.main:app run --host=0.0.0.0 --port=8000 --reload --debug

# production-like local run
../.venv/bin/gunicorn -w 4 -b 0.0.0.0:8000 app.main:app
```

### Quick sanity check
```bash
../.venv/bin/python -m flask --app app.main:app routes
```

## 4) Test Commands (focus: single test)
Current repo state: no committed tests under `backend/tests/` yet.
When adding tests, use `pytest` files named `backend/tests/test_*.py`.

### Run all tests
```bash
../.venv/bin/python -m pytest tests -q
```

### Run a single test file
```bash
../.venv/bin/python -m pytest tests/test_example.py -q
```

### Run one test function (preferred narrow iteration)
```bash
../.venv/bin/python -m pytest tests/test_example.py::test_specific_case -q
```

### Run one class method
```bash
../.venv/bin/python -m pytest tests/test_example.py::TestThing::test_method -q
```

### Filter by keyword
```bash
../.venv/bin/python -m pytest tests -k "login or clinician" -q
```

## 5) Lint and Quality Commands
- No enforced Ruff/Black/isort/Flake8 config is checked in.
- Follow PEP 8 and existing local formatting in each file.
- Webhint configs exist at `/.hintrc` and `/backend/.hintrc`.
- If `hint` is available:
```bash
npx hint backend/app/templates
```
- Python syntax smoke check:
```bash
../.venv/bin/python -m compileall app
```

## 6) Python Code Style
- Use 4-space indentation.
- Use snake_case for functions, methods, variables.
- Use PascalCase for classes (`ModelLoader`, `ResultsRepository`).
- Use UPPER_SNAKE_CASE for module constants (`SAMPLE_RATE`, `MODEL_PATH`).
- Prefer import groups: stdlib, third-party, local; keep grouped and readable.
- Use explicit relative imports inside backend modules (e.g., `from ..core...`).
- Add/keep type hints for new or edited public methods.
- Modern typing syntax is acceptable (`X | None`, `dict[str, Any]`, `Literal`, `TypedDict`).
- Keep return payload shapes stable, especially service/repository dictionaries.

## 7) Architecture and Organization
- Keep route handlers in `main.py` thin: validate -> call service -> respond.
- Put data access in repository classes under `app/db/repositories/`.
- Put business/domain logic in service classes under `app/services/`.
- Prefer extending existing helpers (`FileService`, `BandAnalysisService`, etc.) before adding new utility layers.
- Favor small focused methods over broad mixed-responsibility functions.
- Avoid broad refactors unless requested by the task.

## 8) Naming and API Conventions
- Route function names are snake_case (`get_results`, `api_login`).
- Use JSON error shape `{"error": "..."}` for API failures.
- Keep naming aligned with schema and existing response keys (`subject_code`, `confidence_score`, `predicted_class`).
- Preserve current ADHD label strings expected by frontend templates.

## 9) Error Handling and Logging
- Use loggers from `app/core/logging_config.py`: `get_app_logger`, `get_db_logger`, `get_ml_logger`.
- Log exceptions with `exc_info=True` when stack traces matter.
- In repositories: catch, log, and re-raise by default.
- In Flask routes: return 400 for validation errors, 500 for unexpected failures.
- Avoid silent exception swallowing unless fallback behavior is intentionally required.

## 10) Database and SQL Rules
- Always use parameterized SQL (`%s` placeholders).
- Never build SQL via string concatenation with user input.
- Use `with self.get_connection() as conn:` for transaction lifecycle.
- Use dict cursors (`self.get_dict_cursor(conn)`) when returning JSON-like records.
- Keep schema updates compatible with `backend/app/db/schema.sql`.

## 11) Frontend Template/JS/CSS Conventions
- Frontend is server-rendered HTML + inline vanilla JS + Bootstrap.
- Preserve existing DOM IDs/class names; scripts are tightly coupled to templates.
- Prefer `async/await` + `fetch` patterns already in use.
- Follow nearby quote/spacing style; do not mass-reformat whole templates.
- Keep user-facing error messages concise and actionable.

## 12) Secrets and Generated Files
- Do not commit `.env` secrets or `.secrets/` contents.
- Respect `.gitignore` entries (`exports`, `validation_data`, `logs`, CSV artifacts, etc.).
- Avoid committing large generated ML/data outputs unless explicitly requested.

## 13) Cursor/Copilot Rule Files
Checked locations: `.cursor/rules/`, `.cursorrules`, `.github/copilot-instructions.md`.
Current state: none of these files are present in this repository.

## 14) Agent Finish Checklist
- Run relevant tests (or clearly state that no tests exist yet).
- For backend changes, run route sanity when feasible: `../.venv/bin/python -m flask --app app.main:app routes`.
- Keep changes minimal and scoped.
- Preserve existing architecture (routes -> services -> repositories).
- If you introduce new tooling/commands, document them in this file.
