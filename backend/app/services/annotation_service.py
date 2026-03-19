from __future__ import annotations

from dataclasses import dataclass

from ..core.logging_config import get_app_logger
from ..db.repositories.eeg_annotations import EEGAnnotationsRepository
from .visualization_service import VisualizationService

logger = get_app_logger(__name__)


@dataclass(frozen=True)
class AnnotationPayload:
    band_name: str
    start_time_sec: float
    end_time_sec: float | None
    label: str
    notes: str | None = None
    color: str | None = None


class AnnotationService:
    """Validate and persist EEG waveform annotations."""

    DEFAULT_LABEL = "Annotation"
    MAX_LABEL_LEN = 100
    MAX_NOTES_LEN = 255
    MAX_COLOR_LEN = 20

    def __init__(self, repo: EEGAnnotationsRepository):
        self.repo = repo

    @staticmethod
    def _allowed_bands() -> set[str]:
        return set(VisualizationService.BAND_FILTERS.keys()) | {"segments"}

    def _normalize_payload(self, payload: dict) -> AnnotationPayload:
        band_name = str(payload.get("band_name") or "").strip()
        if band_name not in self._allowed_bands():
            raise ValueError("Invalid band name")

        start_raw = payload.get("start_time_sec")
        end_raw = payload.get("end_time_sec")

        try:
            start_time = float(start_raw)
        except (TypeError, ValueError):
            raise ValueError("start_time_sec must be a number")

        end_time = None
        if end_raw not in (None, ""):
            try:
                end_time = float(end_raw)
            except (TypeError, ValueError):
                raise ValueError("end_time_sec must be a number")

        if start_time < 0:
            raise ValueError("start_time_sec must be >= 0")
        if end_time is not None and end_time < 0:
            raise ValueError("end_time_sec must be >= 0")

        if end_time is not None and end_time < start_time:
            start_time, end_time = end_time, start_time

        label = str(payload.get("label") or "").strip()
        if not label:
            label = self.DEFAULT_LABEL
        if len(label) > self.MAX_LABEL_LEN:
            raise ValueError("label exceeds maximum length")

        notes = payload.get("notes")
        if notes is not None:
            notes = str(notes).strip()
            if len(notes) > self.MAX_NOTES_LEN:
                raise ValueError("notes exceeds maximum length")

        color = payload.get("color")
        if color is not None:
            color = str(color).strip()
            if len(color) > self.MAX_COLOR_LEN:
                raise ValueError("color exceeds maximum length")

        return AnnotationPayload(
            band_name=band_name,
            start_time_sec=start_time,
            end_time_sec=end_time,
            label=label,
            notes=notes,
            color=color,
        )

    def list_annotations(self, result_id: int, band_name: str | None = None) -> list[dict]:
        return self.repo.list_by_result(result_id, band_name=band_name)

    def create_annotation(
        self,
        result_id: int,
        clinician_id: int | None,
        payload: dict,
    ) -> dict:
        normalized = self._normalize_payload(payload)
        annotation_id = self.repo.create(
            result_id=result_id,
            clinician_id=clinician_id,
            band_name=normalized.band_name,
            start_time_sec=normalized.start_time_sec,
            end_time_sec=normalized.end_time_sec,
            label=normalized.label,
            notes=normalized.notes,
            color=normalized.color,
        )
        return self.repo.get_by_id(annotation_id) or {
            "id": annotation_id,
            "result_id": result_id,
            "clinician_id": clinician_id,
            "band_name": normalized.band_name,
            "start_time_sec": normalized.start_time_sec,
            "end_time_sec": normalized.end_time_sec,
            "label": normalized.label,
            "notes": normalized.notes,
            "color": normalized.color,
        }

    def update_annotation(
        self,
        annotation_id: int,
        clinician_id: int | None,
        payload: dict,
    ) -> dict:
        existing = self.repo.get_by_id(annotation_id)
        if not existing:
            raise LookupError("Annotation not found")
        if clinician_id is not None and existing.get("clinician_id") not in (None, clinician_id):
            raise PermissionError("Not authorized to edit this annotation")

        merged = {
            "band_name": existing.get("band_name"),
            "start_time_sec": payload.get("start_time_sec", existing.get("start_time_sec")),
            "end_time_sec": payload.get("end_time_sec", existing.get("end_time_sec")),
            "label": payload.get("label", existing.get("label")),
            "notes": payload.get("notes", existing.get("notes")),
            "color": payload.get("color", existing.get("color")),
        }
        normalized = self._normalize_payload(merged)
        updated = self.repo.update(
            annotation_id=annotation_id,
            start_time_sec=normalized.start_time_sec,
            end_time_sec=normalized.end_time_sec,
            label=normalized.label,
            notes=normalized.notes,
            color=normalized.color,
        )
        if not updated:
            raise LookupError("Annotation not found")
        return updated

    def delete_annotation(self, annotation_id: int, clinician_id: int | None) -> None:
        existing = self.repo.get_by_id(annotation_id)
        if not existing:
            raise LookupError("Annotation not found")
        if clinician_id is not None and existing.get("clinician_id") not in (None, clinician_id):
            raise PermissionError("Not authorized to delete this annotation")
        self.repo.delete(annotation_id)
