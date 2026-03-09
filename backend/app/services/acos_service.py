from __future__ import annotations

from typing import TypedDict


class AcosSubscaleResult(TypedDict):
    score: int
    average_score: float
    severity: str


class AcosResult(TypedDict):
    total_score: int
    average_score: float
    severity: str
    subscales: dict[str, AcosSubscaleResult]


class AcosService:
    """Compute ADHD Clinical Outcome Scale (ACOS-C) scores and descriptors."""

    SUBSCALES: dict[str, tuple[int, ...]] = {
        "attention_and_functional_difficulties": (2, 6, 10, 14, 15),
        "hyperactivity_impulsivity_emotional_dysregulation": (1, 3, 5, 7),
        "cooccurring_mental_health_problems": (11, 12, 13),
        "risk_behaviours_interpersonal_problems": (4, 8, 9),
    }

    @staticmethod
    def severity_for_average(average_score: float) -> str:
        if average_score < 0.5:
            return "No Problem"
        if average_score < 1.5:
            return "Minor"
        if average_score < 2.5:
            return "Mild"
        if average_score < 3.5:
            return "Moderately Severe"
        if average_score < 4.5:
            return "Severe"
        return "Very Severe"

    def compute(self, item_scores: dict[int, int]) -> AcosResult:
        if len(item_scores) != 15:
            raise ValueError("ACOS requires 15 item scores")

        for item_index in range(1, 16):
            if item_index not in item_scores:
                raise ValueError(f"Missing ACOS item {item_index}")
            value = item_scores[item_index]
            if not isinstance(value, int) or not (0 <= value <= 5):
                raise ValueError(f"ACOS item {item_index} must be an integer between 0 and 5")

        total_score = sum(item_scores.values())
        average_score = total_score / 15

        subscale_results: dict[str, AcosSubscaleResult] = {}
        for subscale_key, items in self.SUBSCALES.items():
            subscale_score = sum(item_scores[item] for item in items)
            subscale_average = subscale_score / len(items)
            subscale_results[subscale_key] = {
                "score": subscale_score,
                "average_score": round(subscale_average, 3),
                "severity": self.severity_for_average(subscale_average),
            }

        return {
            "total_score": total_score,
            "average_score": round(average_score, 3),
            "severity": self.severity_for_average(average_score),
            "subscales": subscale_results,
        }
