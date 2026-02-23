# Testing Patterns

**Analysis Date:** 2026-02-23

## Test Framework

**Runner:**
- Not detected (no `pytest.ini`, `tox.ini`, `pyproject.toml`, or `unittest` test modules in repository root or `backend/`).
- Config: Not detected.

**Assertion Library:**
- Not detected for automated tests.

**Run Commands:**
```bash
Not applicable - no automated test command configured in repository files.
Not applicable - no watch mode configured.
Not applicable - no coverage command configured.
```

## Test File Organization

**Location:**
- No automated test directories detected (`tests/`, `backend/tests/`, or `test_*.py` patterns are absent).
- Manual validation scripts live under `backend/scripts/` (for example `backend/scripts/validation_data_script.py`).

**Naming:**
- Automated test naming pattern not detected.
- Validation scripts use descriptive `snake_case.py` filenames in `backend/scripts/frequency_domain_heatmap.py` and `backend/scripts/export_individual_eegs.py`.

**Structure:**
```
backend/scripts/*.py  # manual validation and data-processing checks
backend/notebooks/*.ipynb  # exploratory model experiments, not test suites
```

## Test Structure

**Suite Organization:**
```python
# Manual-execution pattern used for verification scripts
if __name__ == "__main__":
    df = pd.read_csv("./adhdata.csv")
    # process and export validation artifacts
```

**Patterns:**
- Setup pattern: scripts build local output directories with `Path(...).mkdir(...)` in `backend/scripts/validation_data_script.py` and `backend/scripts/frequency_domain_heatmap.py`.
- Teardown pattern: none standardized; scripts generally close matplotlib figures (`plt.close(...)`) in `backend/scripts/validation_data_script.py` and `backend/scripts/frequency_domain_heatmap.py`.
- Assertion pattern: little explicit assertion usage; runtime validation mostly uses exceptions (`raise ValueError`) in `backend/app/services/file_service.py` and `backend/app/services/eeg_service.py`.

## Mocking

**Framework:**
- Not detected for automated tests.

**Patterns:**
```python
# Runtime mock-data helper, not a test mock framework
subject_data = MockDataGenerator.generate_subject_data()
recording_data = MockDataGenerator.generate_recording_data(file_name)
```

**What to Mock:**
- For future automated tests, mock external boundaries first: database access in `backend/app/db/connection.py`, model loading/inference in `backend/app/ml/model_loader.py`, and file input (`FileStorage`) in `backend/app/services/file_service.py`.

**What NOT to Mock:**
- Keep pure data-shaping logic real where practical, such as name parsing in `backend/app/services/clinician_service.py` and band/ratio transformations in `backend/app/services/eeg_service.py`.

## Fixtures and Factories

**Test Data:**
```python
# Existing factory-like generator used by application code
MockDataGenerator.generate_subject_data()
MockDataGenerator.generate_recording_data(file_name)
```

**Location:**
- `backend/app/utils/mock_data.py` contains reusable synthetic subject/recording generators.

## Coverage

**Requirements:** None enforced by repository configuration.

**View Coverage:**
```bash
Not applicable - coverage tooling not configured.
```

## Test Types

**Unit Tests:**
- Not used (no unit test files detected).

**Integration Tests:**
- Not used as a formal suite; integration is validated manually by running the app (`backend/Dockerfile` development command) and using API/UI flows in `backend/app/main.py` plus templates in `backend/app/templates/index.html`.

**E2E Tests:**
- Not used (no Playwright/Cypress/Selenium configuration detected).

## Common Patterns

**Async Testing:**
```python
# No async test pattern detected.
# Async behavior appears only in frontend JS via fetch() calls in
# backend/app/templates/index.html and backend/app/templates/login.html.
```

**Error Testing:**
```python
# Validation/error behavior currently exercised through runtime checks
if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found in file.")
```

---

*Testing analysis: 2026-02-23*
