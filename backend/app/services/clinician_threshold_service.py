from typing import Any

from ..db.repositories.clinician_thresholds import ClinicianThresholdsRepository


class ClinicianThresholdService:
    ALLOWED_SUBTYPES = {
        "inattentive": "Inattentive",
        "hyperactive_impulsive": "Hyperactive-Impulsive",
        "combined": "Combined",
    }
    ALLOWED_BANDS = ("delta", "theta", "alpha", "beta", "gamma")

    def __init__(self, repo: ClinicianThresholdsRepository):
        self.repo = repo

    def get_thresholds(self, clinician_id: int) -> dict[str, dict[str, dict[str, float]]]:
        rows = self.repo.get_by_clinician(clinician_id)
        thresholds: dict[str, dict[str, dict[str, float]]] = {}
        for row in rows:
            subtype = row["adhd_subtype"]
            band = row["band"]
            thresholds.setdefault(subtype, {})[band] = {
                "min": float(row["min_value"]),
                "max": float(row["max_value"]),
            }
        return thresholds

    def replace_thresholds(self, clinician_id: int, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Invalid configuration payload")

        rows: list[dict[str, Any]] = []
        for subtype, bands in payload.items():
            if subtype not in self.ALLOWED_SUBTYPES:
                raise ValueError(f"Invalid ADHD subtype: {subtype}")
            if not isinstance(bands, dict):
                raise ValueError("Thresholds must include band settings")
            for band, values in bands.items():
                if band not in self.ALLOWED_BANDS:
                    raise ValueError(f"Invalid band: {band}")
                if not isinstance(values, dict):
                    raise ValueError("Band threshold values must be an object")
                min_value = values.get("min")
                max_value = values.get("max")
                min_value = self._to_float(min_value, "min")
                max_value = self._to_float(max_value, "max")
                if min_value < 0 or max_value > 1 or min_value > max_value:
                    raise ValueError(
                        f"Invalid range for {subtype} {band}: {min_value}-{max_value}"
                    )
                rows.append(
                    {
                        "adhd_subtype": subtype,
                        "band": band,
                        "min_value": min_value,
                        "max_value": max_value,
                    }
                )

        self.repo.replace_for_clinician(clinician_id, rows)

    def get_default_thresholds(self) -> dict[str, dict[str, dict[str, float]]]:
        defaults: dict[str, dict[str, dict[str, float]]] = {}
        for subtype in self.ALLOWED_SUBTYPES:
            defaults[subtype] = {
                band: {"min": 0.0, "max": 1.0} for band in self.ALLOWED_BANDS
            }
        return defaults

    @staticmethod
    def _to_float(value: Any, label: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {label} value")
        if not (number == number) or number in (float("inf"), float("-inf")):
            raise ValueError(f"Invalid {label} value")
        return number
