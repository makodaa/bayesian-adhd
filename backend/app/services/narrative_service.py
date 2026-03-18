"""
Narrative interpretation service for EEG analysis results.

Generates a short clinical paragraph framing findings in terms of
cortical arousal and attention regulation, based on computed band
powers and spectral ratios.
"""

from ..core.logging_config import get_app_logger

logger = get_app_logger(__name__)

# Thresholds used to characterise arousal state
_TBR_HYPO_THRESHOLD = 3.0      # theta/beta ratio above this → hypoarousal
_TBR_HYPER_THRESHOLD = 1.5     # theta/beta ratio below this (with high beta) → hyperarousal
_THETA_HYPO_THRESHOLD = 0.35   # relative theta above this → hypoarousal signal
_BETA_HYPER_THRESHOLD = 0.25   # relative beta above this (with low TBR) → hyperarousal signal


class NarrativeService:
    """
    Produce a short narrative interpretation of EEG findings in terms of
    cortical arousal and attention regulation.

    All inputs are derived from values already computed and stored by
    BandAnalysisService; no model inference is performed here.
    """

    def generate_arousal_narrative(
        self,
        predicted_class: str,
        confidence_score: float,
        avg_relative_power: dict[str, float],
        band_ratios: dict[str, float],
    ) -> str:
        """
        Generate a 3-sentence narrative paragraph.

        Parameters
        ----------
        predicted_class : str
            The model's predicted class label (e.g. "ADHD Inattentive (ADHD-I)").
        confidence_score : float
            Model confidence in [0, 1].
        avg_relative_power : dict[str, float]
            Average relative band power keyed by band name
            (e.g. {"theta": 0.38, "beta": 0.18, ...}).
            Values should be fractions (0–1), not percentages.
        band_ratios : dict[str, float]
            Computed spectral ratios, e.g. {"theta_beta_ratio": 3.2, ...}.

        Returns
        -------
        str
            Plain-text narrative suitable for display or PDF embedding.
        """
        logger.debug(
            "Generating arousal narrative for class=%s confidence=%.3f",
            predicted_class,
            confidence_score,
        )

        theta = avg_relative_power.get("theta", 0.0)
        beta = avg_relative_power.get("beta", 0.0)
        alpha = avg_relative_power.get("alpha", 0.0)
        tbr = band_ratios.get("theta_beta_ratio", 0.0)

        theta_pct = theta * 100
        beta_pct = beta * 100
        alpha_pct = alpha * 100

        is_adhd = self._is_adhd(predicted_class)

        # ── Sentence 1: spectral overview ──────────────────────────────────
        s1 = (
            f"EEG spectral analysis shows relative theta power of {theta_pct:.1f}%, "
            f"relative beta power of {beta_pct:.1f}%, "
            f"relative alpha power of {alpha_pct:.1f}%, "
            f"and a theta/beta ratio (TBR) of {tbr:.2f}."
        )

        # ── Sentence 2: arousal characterisation ───────────────────────────
        s2 = self._arousal_sentence(theta, beta, tbr)

        # ── Sentence 3: attention regulation ───────────────────────────────
        s3 = self._attention_sentence(is_adhd, tbr)

        narrative = " ".join([s1, s2, s3])
        logger.debug("Narrative generated (%d characters)", len(narrative))
        return narrative

    # ── internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _is_adhd(predicted_class: str) -> bool:
        label = str(predicted_class).lower()
        return "non-adhd" not in label and "adhd" in label

    @staticmethod
    def _arousal_sentence(theta: float, beta: float, tbr: float) -> str:
        """Characterise the cortical arousal state from spectral values."""
        hypoarousal = tbr > _TBR_HYPO_THRESHOLD or theta > _THETA_HYPO_THRESHOLD
        hyperarousal = tbr < _TBR_HYPER_THRESHOLD and beta > _BETA_HYPER_THRESHOLD

        if hypoarousal:
            return (
                "The elevated theta activity and theta/beta ratio are consistent "
                "with a cortical hypoarousal pattern, suggesting reduced frontal "
                "activation and decreased inhibitory tone."
            )
        if hyperarousal:
            return (
                "The relatively high beta power and low theta/beta ratio suggest "
                "a cortical hyperarousal or heightened activation pattern, which "
                "may be associated with anxiety, hypervigilance, or stimulant effects."
            )
        return (
            "The theta/beta ratio and band power distribution do not clearly "
            "indicate cortical arousal dysregulation within this recording."
        )

    @staticmethod
    def _attention_sentence(is_adhd: bool, tbr: float) -> str:
        """Characterise attention regulation from arousal pattern and classification."""
        elevated_tbr = tbr > _TBR_HYPO_THRESHOLD

        if is_adhd and elevated_tbr:
            return (
                "This profile is consistent with reduced attentional gating and "
                "impaired frontally mediated inhibitory control."
            )

        if is_adhd and not elevated_tbr:
            return (
                "The theta/beta ratio does not fall in the typically elevated range; "
                "other spectral and temporal features may have contributed to this classification."
            )

        if not is_adhd and elevated_tbr:
            return (
                "The elevated theta/beta ratio warrants clinical consideration, as this "
                "pattern can occasionally appear in other neurodevelopmental or attentional presentations."
            )

        # Non-ADHD, normal TBR
        return (
            "The attention regulation markers in this recording do not indicate "
            "spectral patterns typically associated with ADHD-related cortical dysregulation."
        )
