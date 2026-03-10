from __future__ import annotations

import math
from typing import Literal, TypedDict


VanderbiltScaleType = Literal["initial", "follow_up"]


class VanderbiltDomainResult(TypedDict):
    symptom_positive_count: int
    threshold_met: bool


class VanderbiltResult(TypedDict):
    scale_type: VanderbiltScaleType
    inattentive_count: int
    hyperactive_impulsive_count: int
    performance_items: dict[str, int]
    performance_impairment_count: int
    adhd_inattentive_criteria_met: bool
    adhd_hyperactive_impulsive_criteria_met: bool
    adhd_combined_criteria_met: bool
    criteria_outcome: str
    interpretation: str
    normative: dict[str, dict[str, float | str]]
    domains: dict[str, VanderbiltDomainResult]


class VanderbiltService:
    """Compute ADHD-core Vanderbilt symptom and impairment outcomes."""

    INITIAL_SYMPTOM_ITEM_COUNT = 18
    FOLLOW_UP_SYMPTOM_ITEM_COUNT = 18
    REQUIRED_PERFORMANCE_KEYS = (
        "school_performance",
        "reading",
        "writing",
        "mathematics",
        "relationship_parents",
        "relationship_siblings",
        "relationship_peers",
        "classroom_behavior",
        "assignment_completion",
        "organizational_skills",
    )

    NORMATIVE_REFERENCE = {
        "inattentive": {"mean": 8.1, "sd": 5.70},
        "hyperactive_impulsive": {"mean": 7.5, "sd": 6.04},
        "combined": {"mean": 15.6, "sd": 10.96},
    }

    @staticmethod
    def symptom_positive(value: int) -> bool:
        return value >= 2

    @staticmethod
    def performance_impaired(value: int) -> bool:
        return value >= 4

    def _validate_symptoms(
        self,
        symptom_scores: dict[int, int],
        scale_type: VanderbiltScaleType,
    ) -> None:
        expected_count = (
            self.INITIAL_SYMPTOM_ITEM_COUNT
            if scale_type == "initial"
            else self.FOLLOW_UP_SYMPTOM_ITEM_COUNT
        )

        if len(symptom_scores) != expected_count:
            raise ValueError(
                f"Vanderbilt {scale_type} requires {expected_count} symptom item scores"
            )

        for item_index in range(1, expected_count + 1):
            if item_index not in symptom_scores:
                raise ValueError(f"Missing Vanderbilt symptom item {item_index}")
            value = symptom_scores[item_index]
            if not isinstance(value, int) or not (0 <= value <= 3):
                raise ValueError(
                    f"Vanderbilt symptom item {item_index} must be an integer between 0 and 3"
                )

    def _validate_performance(self, performance_scores: dict[str, int]) -> None:
        missing = [
            key for key in self.REQUIRED_PERFORMANCE_KEYS if key not in performance_scores
        ]
        if missing:
            raise ValueError(
                "Missing Vanderbilt performance items: " + ", ".join(sorted(missing))
            )

        for key in self.REQUIRED_PERFORMANCE_KEYS:
            value = performance_scores[key]
            if not isinstance(value, int) or not (1 <= value <= 5):
                raise ValueError(
                    f"Vanderbilt performance item '{key}' must be an integer between 1 and 5"
                )

    @staticmethod
    def _normal_cdf(z_score: float) -> float:
        return 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))

    @staticmethod
    def _normative_descriptor(percentile: float) -> str:
        if percentile >= 95:
            return "Very elevated"
        if percentile >= 85:
            return "Elevated"
        if percentile < 16:
            return "Lower than normative mean range"
        return "Typical range"

    def _build_normative_domain(self, raw_score: int, mean: float, sd: float) -> dict[str, float | str]:
        z_score = (raw_score - mean) / sd
        percentile = self._normal_cdf(z_score) * 100
        return {
            "raw_score": float(raw_score),
            "mean": mean,
            "sd": sd,
            "z_score": round(z_score, 2),
            "percentile": round(percentile, 1),
            "descriptor": self._normative_descriptor(percentile),
        }

    def compute(
        self,
        symptom_scores: dict[int, int],
        performance_scores: dict[str, int],
        scale_type: VanderbiltScaleType,
    ) -> VanderbiltResult:
        self._validate_symptoms(symptom_scores, scale_type)
        self._validate_performance(performance_scores)

        inattentive_count = sum(
            1 for i in range(1, 10) if self.symptom_positive(symptom_scores[i])
        )
        hyperactive_impulsive_count = sum(
            1 for i in range(10, 19) if self.symptom_positive(symptom_scores[i])
        )

        performance_impairment_count = sum(
            1
            for key in self.REQUIRED_PERFORMANCE_KEYS
            if self.performance_impaired(performance_scores[key])
        )

        inattentive_met = inattentive_count >= 6 and performance_impairment_count >= 1
        hyperactive_met = (
            hyperactive_impulsive_count >= 6 and performance_impairment_count >= 1
        )
        combined_met = inattentive_met and hyperactive_met
        combined_score = inattentive_count + hyperactive_impulsive_count

        normative = {
            "inattentive": self._build_normative_domain(
                inattentive_count,
                self.NORMATIVE_REFERENCE["inattentive"]["mean"],
                self.NORMATIVE_REFERENCE["inattentive"]["sd"],
            ),
            "hyperactive_impulsive": self._build_normative_domain(
                hyperactive_impulsive_count,
                self.NORMATIVE_REFERENCE["hyperactive_impulsive"]["mean"],
                self.NORMATIVE_REFERENCE["hyperactive_impulsive"]["sd"],
            ),
            "combined": self._build_normative_domain(
                combined_score,
                self.NORMATIVE_REFERENCE["combined"]["mean"],
                self.NORMATIVE_REFERENCE["combined"]["sd"],
            ),
        }

        if combined_met:
            criteria_outcome = "ADHD Combined Presentation Criteria Met"
            interpretation = (
                "Findings are consistent with ADHD combined presentation criteria on the "
                "Vanderbilt parent scale and should be interpreted alongside full clinical assessment."
            )
        elif inattentive_met:
            criteria_outcome = "ADHD Inattentive Presentation Criteria Met"
            interpretation = (
                "Findings are consistent with ADHD inattentive presentation criteria on the "
                "Vanderbilt parent scale and should be interpreted alongside full clinical assessment."
            )
        elif hyperactive_met:
            criteria_outcome = "ADHD Hyperactive/Impulsive Presentation Criteria Met"
            interpretation = (
                "Findings are consistent with ADHD hyperactive/impulsive presentation criteria on the "
                "Vanderbilt parent scale and should be interpreted alongside full clinical assessment."
            )
        else:
            criteria_outcome = "ADHD Core Criteria Not Met"
            interpretation = (
                "Vanderbilt ADHD-core criteria are not fully met in this response set. "
                "Findings remain supportive only and do not replace clinical diagnosis."
            )

        return {
            "scale_type": scale_type,
            "inattentive_count": inattentive_count,
            "hyperactive_impulsive_count": hyperactive_impulsive_count,
            "performance_items": {
                key: performance_scores[key] for key in self.REQUIRED_PERFORMANCE_KEYS
            },
            "performance_impairment_count": performance_impairment_count,
            "adhd_inattentive_criteria_met": inattentive_met,
            "adhd_hyperactive_impulsive_criteria_met": hyperactive_met,
            "adhd_combined_criteria_met": combined_met,
            "criteria_outcome": criteria_outcome,
            "interpretation": interpretation,
            "normative": normative,
            "domains": {
                "inattentive": {
                    "symptom_positive_count": inattentive_count,
                    "threshold_met": inattentive_count >= 6,
                },
                "hyperactive_impulsive": {
                    "symptom_positive_count": hyperactive_impulsive_count,
                    "threshold_met": hyperactive_impulsive_count >= 6,
                },
                "performance": {
                    "symptom_positive_count": performance_impairment_count,
                    "threshold_met": performance_impairment_count >= 1,
                },
            },
        }
