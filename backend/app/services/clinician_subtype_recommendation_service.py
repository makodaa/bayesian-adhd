from typing import Any

from ..db.repositories.clinician_subtype_recommendations import (
    ClinicianSubtypeRecommendationsRepository,
)


class ClinicianSubtypeRecommendationService:
    ALLOWED_SUBTYPES = {
        "inattentive": "Inattentive",
        "hyperactive_impulsive": "Hyperactive-Impulsive",
        "combined": "Combined",
    }
    ALLOWED_TRIGGERS = ("consistent_subtype",)

    def __init__(self, repo: ClinicianSubtypeRecommendationsRepository):
        self.repo = repo

    def get_recommendations(self, clinician_id: int) -> dict[str, dict[str, str]]:
        rows = self.repo.get_by_clinician(clinician_id)
        recommendations: dict[str, dict[str, str]] = {}
        for row in rows:
            subtype = row["adhd_subtype"]
            trigger = row["trigger_key"]
            recommendations.setdefault(subtype, {})[trigger] = (
                row.get("recommendation_text") or ""
            )
        return recommendations

    def replace_recommendations(self, clinician_id: int, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Invalid subtype recommendation payload")

        rows: list[dict[str, Any]] = []
        for subtype, triggers in payload.items():
            if subtype not in self.ALLOWED_SUBTYPES:
                raise ValueError(f"Invalid ADHD subtype: {subtype}")
            if not isinstance(triggers, dict):
                raise ValueError("Subtype recommendations must include trigger settings")
            for trigger, text in triggers.items():
                if trigger not in self.ALLOWED_TRIGGERS:
                    raise ValueError(f"Invalid trigger: {trigger}")
                if text is None:
                    recommendation = ""
                else:
                    recommendation = str(text).strip()
                if len(recommendation) > 800:
                    raise ValueError(
                        f"Recommendation too long for {subtype} {trigger}"
                    )
                if not recommendation:
                    continue
                rows.append(
                    {
                        "adhd_subtype": subtype,
                        "trigger_key": trigger,
                        "recommendation_text": recommendation,
                    }
                )

        self.repo.replace_for_clinician(clinician_id, rows)

    def get_default_recommendations(self) -> dict[str, dict[str, str]]:
        defaults: dict[str, dict[str, str]] = {}
        for subtype in self.ALLOWED_SUBTYPES:
            defaults[subtype] = {trigger: "" for trigger in self.ALLOWED_TRIGGERS}
        return defaults

    def get_recommendation_matrix(self, clinician_id: int) -> dict[str, dict[str, str]]:
        recommendations = self.get_recommendations(clinician_id)
        defaults = self.get_default_recommendations()
        for subtype, triggers in defaults.items():
            defaults[subtype].update(recommendations.get(subtype, {}))
        return defaults
