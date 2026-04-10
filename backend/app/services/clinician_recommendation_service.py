from typing import Any

from ..db.repositories.clinician_recommendations import (
    ClinicianRecommendationsRepository,
)


class ClinicianRecommendationService:
    ALLOWED_SUBTYPES = {
        "inattentive": "Inattentive",
        "hyperactive_impulsive": "Hyperactive-Impulsive",
        "combined": "Combined",
    }
    ALLOWED_BANDS = ("alpha", "beta", "theta")
    ALLOWED_STATES = ("decreased", "elevated")

    def __init__(self, repo: ClinicianRecommendationsRepository):
        self.repo = repo

    def get_recommendations(
        self, clinician_id: int
    ) -> dict[str, dict[str, dict[str, str]]]:
        rows = self.repo.get_by_clinician(clinician_id)
        recommendations: dict[str, dict[str, dict[str, str]]] = {}
        for row in rows:
            subtype = row["adhd_subtype"]
            band = row["band"]
            state = row["band_state"]
            recommendations.setdefault(subtype, {}).setdefault(band, {})[state] = (
                row.get("recommendation_text") or ""
            )
        return recommendations

    def replace_recommendations(self, clinician_id: int, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Invalid recommendation payload")

        rows: list[dict[str, Any]] = []
        for subtype, bands in payload.items():
            if subtype not in self.ALLOWED_SUBTYPES:
                raise ValueError(f"Invalid ADHD subtype: {subtype}")
            if not isinstance(bands, dict):
                raise ValueError("Recommendations must include band settings")
            for band, states in bands.items():
                if band not in self.ALLOWED_BANDS:
                    raise ValueError(f"Invalid band: {band}")
                if not isinstance(states, dict):
                    raise ValueError("Band recommendations must be an object")
                for state, text in states.items():
                    if state not in self.ALLOWED_STATES:
                        raise ValueError(f"Invalid band state: {state}")
                    if text is None:
                        recommendation = ""
                    else:
                        recommendation = str(text).strip()
                    if len(recommendation) > 800:
                        raise ValueError(
                            f"Recommendation too long for {subtype} {band} {state}"
                        )
                    if not recommendation:
                        continue
                    rows.append(
                        {
                            "adhd_subtype": subtype,
                            "band": band,
                            "band_state": state,
                            "recommendation_text": recommendation,
                        }
                    )

        self.repo.replace_for_clinician(clinician_id, rows)

    def get_default_recommendations(self) -> dict[str, dict[str, dict[str, str]]]:
        defaults: dict[str, dict[str, dict[str, str]]] = {}
        for subtype in self.ALLOWED_SUBTYPES:
            defaults[subtype] = {}
            for band in self.ALLOWED_BANDS:
                defaults[subtype][band] = {state: "" for state in self.ALLOWED_STATES}
        return defaults

    def get_recommendation_matrix(
        self, clinician_id: int
    ) -> dict[str, dict[str, dict[str, str]]]:
        recommendations = self.get_recommendations(clinician_id)
        defaults = self.get_default_recommendations()
        for subtype, bands in defaults.items():
            for band, states in bands.items():
                defaults[subtype][band].update(recommendations.get(subtype, {}).get(band, {}))
        return defaults
