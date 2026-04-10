"""
Centralised input validation helpers.

All functions raise ``ValueError`` with a user-facing message on failure.
They are pure (no DB access, no Flask context) so they can be called from
both route handlers and service methods.
"""

from __future__ import annotations

import re
from datetime import date

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_GENDERS = ("Male", "Female", "Other", "Prefer not to say")

_NAME_RE = re.compile(r"^[A-Za-z\s'\-]+$")
_OCCUPATION_RE = re.compile(r"^[A-Za-z\s'\-]+$")

TEXT_FIELD_MAX = 500       # characters for free-text clinical fields
NAME_FIELD_MAX = 50        # characters for name / occupation fields
SUBJECT_CODE_MAX = 60      # characters for subject_code

# Password strength requirements
_PW_MIN_LEN = 8
_PW_RE_UPPER = re.compile(r"[A-Z]")
_PW_RE_LOWER = re.compile(r"[a-z]")
_PW_RE_DIGIT = re.compile(r"[0-9]")
_PW_RE_SPECIAL = re.compile(r"[^A-Za-z0-9]")


# ---------------------------------------------------------------------------
# Subject / recording helpers
# ---------------------------------------------------------------------------

def validate_subject_code(value: str | None) -> str:
    """Strip whitespace and enforce max-length on a subject code.

    Raises ``ValueError`` if the value is empty or exceeds ``SUBJECT_CODE_MAX``.
    """
    if not value:
        raise ValueError("Subject code is required.")
    stripped = value.strip()
    if not stripped:
        raise ValueError("Subject code is required.")
    if len(stripped) > SUBJECT_CODE_MAX:
        raise ValueError(
            f"Subject code must be {SUBJECT_CODE_MAX} characters or fewer."
        )
    return stripped


def validate_age(value: str | int | None) -> int:
    """Parse and range-check age.

    Raises ``ValueError`` if the value is missing, non-integer, or outside 1–120.
    """
    if value is None or value == "":
        raise ValueError("Age is required.")
    try:
        age_int = int(value)
    except (TypeError, ValueError):
        raise ValueError("Age must be a whole number.")
    if not (1 <= age_int <= 120):
        raise ValueError("Age must be between 1 and 120.")
    return age_int


def validate_gender(value: str | None) -> str:
    """Ensure gender is one of the accepted enum values.

    Raises ``ValueError`` if the value is missing or not in ``VALID_GENDERS``.
    """
    if not value:
        raise ValueError("Gender is required.")
    stripped = value.strip()
    if stripped not in VALID_GENDERS:
        raise ValueError(
            f"Gender must be one of: {', '.join(VALID_GENDERS)}."
        )
    return stripped


def validate_date_of_birth(value: str | None) -> date:
    """Validate an ISO date-of-birth string.

    Expected format: YYYY-MM-DD.
    Raises ``ValueError`` if missing, invalid, in the future, or age not between 1–100.
    """
    if not value:
        raise ValueError("Date of birth is required.")
    stripped = value.strip()
    if not stripped:
        raise ValueError("Date of birth is required.")
    try:
        parsed = date.fromisoformat(stripped)
    except ValueError as exc:
        raise ValueError("Date of birth must be in YYYY-MM-DD format.") from exc
    if parsed > date.today():
        raise ValueError("Date of birth cannot be in the future.")
    
    # Validate that age is between 1 and 100 years
    age = compute_age_from_dob(parsed)
    if not (1 <= age <= 100):
        raise ValueError("Subject age must be between 1 and 100 years.")
    
    return parsed


def compute_age_from_dob(dob: date, reference_date: date | None = None) -> int:
    """Compute age in years from DOB and a reference date."""
    ref = reference_date or date.today()
    years = ref.year - dob.year
    if (ref.month, ref.day) < (dob.month, dob.day):
        years -= 1
    return max(years, 0)


def validate_sleep_hours(value: str | float | None, required: bool = True) -> float | None:
    """Parse and range-check sleep hours.

    Returns ``None`` if the value is absent/blank and ``required`` is False.
    Raises ``ValueError`` if the value is missing (when required) or non-numeric
    or out of range 0–99.99.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ValueError("Sleep hours are required.")
        return None
    try:
        hours = float(value)
    except (TypeError, ValueError):
        raise ValueError("Sleep hours must be a number.")
    if not (0 <= hours <= 99.99):
        raise ValueError("Sleep hours must be between 0 and 99.99.")
    return hours


def validate_hours_ago(
    value: str | float | None,
    field_label: str,
    min_value: float = 0.0,
    max_value: float = 99.99,
    required: bool = True,
) -> float | None:
    """Parse and range-check an hours-ago field.

    Returns ``None`` if the value is absent/blank and ``required`` is False.
    Raises ``ValueError`` if the value is missing (when required) or non-numeric
    or out of range.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ValueError(f"{field_label} is required.")
        return None
    try:
        hours = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_label} must be a number.")
    if not (min_value <= hours <= max_value):
        raise ValueError(
            f"{field_label} must be between {min_value:g} and {max_value:g}."
        )
    return hours


def validate_text_field(
    value: str | None,
    field_label: str = "This field",
    max_len: int = TEXT_FIELD_MAX,
) -> str | None:
    """Trim and length-check an optional free-text field.

    Returns ``None`` if the value is absent/blank.
    Raises ``ValueError`` if the trimmed value exceeds ``max_len``.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) > max_len:
        raise ValueError(
            f"{field_label} must be {max_len} characters or fewer "
            f"({len(stripped)} provided)."
        )
    return stripped


# ---------------------------------------------------------------------------
# Clinician helpers
# ---------------------------------------------------------------------------

def validate_name_field(value: str | None, field_label: str) -> str:
    """Validate a required name field (first name, last name).

    Rules:
    - Non-empty after stripping
    - At most ``NAME_FIELD_MAX`` characters
    - Only letters, spaces, hyphens, apostrophes

    Raises ``ValueError`` with a user-facing message on any violation.
    """
    if not value:
        raise ValueError(f"{field_label} is required.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_label} is required.")
    if len(stripped) > NAME_FIELD_MAX:
        raise ValueError(
            f"{field_label} must be {NAME_FIELD_MAX} characters or fewer."
        )
    if not _NAME_RE.match(stripped):
        raise ValueError(
            f"{field_label} may only contain letters, spaces, hyphens, and apostrophes."
        )
    return stripped


def validate_occupation(value: str | None) -> str:
    """Validate a required occupation field.

    Same character rules as name fields.
    Raises ``ValueError`` with a user-facing message on any violation.
    """
    if not value:
        raise ValueError("Occupation is required.")
    stripped = value.strip()
    if not stripped:
        raise ValueError("Occupation is required.")
    if len(stripped) > NAME_FIELD_MAX:
        raise ValueError(
            f"Occupation must be {NAME_FIELD_MAX} characters or fewer."
        )
    if not _OCCUPATION_RE.match(stripped):
        raise ValueError(
            "Occupation may only contain letters, spaces, hyphens, and apostrophes."
        )
    return stripped


def validate_password_strength(value: str | None) -> None:
    """Enforce password strength requirements.

    Requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character

    Raises ``ValueError`` listing all unmet requirements.
    """
    if not value:
        raise ValueError("Password is required.")

    issues: list[str] = []

    if len(value) < _PW_MIN_LEN:
        issues.append(f"at least {_PW_MIN_LEN} characters")
    if not _PW_RE_UPPER.search(value):
        issues.append("at least one uppercase letter")
    if not _PW_RE_LOWER.search(value):
        issues.append("at least one lowercase letter")
    if not _PW_RE_DIGIT.search(value):
        issues.append("at least one number")
    if not _PW_RE_SPECIAL.search(value):
        issues.append("at least one special character")

    if issues:
        raise ValueError(
            "Password must contain " + ", ".join(issues) + "."
        )
