from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)


class EEGAnnotationsRepository(BaseRepository):
    """Repository for EEG waveform annotations."""

    def list_by_result(self, result_id: int, band_name: str | None = None) -> list[dict]:
        logger.debug(
            "Fetching annotations for result %s (band=%s)",
            result_id,
            band_name,
        )
        query = """
        SELECT a.id,
               a.result_id,
               a.clinician_id,
               a.band_name,
               a.start_time_sec,
               a.end_time_sec,
               a.lane_start,
               a.lane_end,
               a.label,
               a.notes,
               a.color,
               a.created_at,
               a.updated_at,
               TRIM(CONCAT_WS(' ', c.first_name, c.last_name)) AS clinician_name
        FROM eeg_annotations a
        LEFT JOIN clinicians c ON a.clinician_id = c.id
        WHERE a.result_id = %s
        """
        params: list[object] = [result_id]
        if band_name:
            query += " AND a.band_name = %s"
            params.append(band_name)
        query += " ORDER BY a.start_time_sec, a.id;"

        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, tuple(params))
                return cursor.fetchall()
        except Exception as exc:
            logger.error(
                "Failed to fetch annotations for result %s: %s",
                result_id,
                exc,
                exc_info=True,
            )
            raise

    def get_by_id(self, annotation_id: int) -> dict | None:
        logger.debug("Fetching annotation %s", annotation_id)
        query = """
        SELECT a.id,
               a.result_id,
               a.clinician_id,
               a.band_name,
               a.start_time_sec,
               a.end_time_sec,
               a.lane_start,
               a.lane_end,
               a.label,
               a.notes,
               a.color,
               a.created_at,
               a.updated_at,
               TRIM(CONCAT_WS(' ', c.first_name, c.last_name)) AS clinician_name
        FROM eeg_annotations a
        LEFT JOIN clinicians c ON a.clinician_id = c.id
        WHERE a.id = %s;
        """
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (annotation_id,))
                return cursor.fetchone()
        except Exception as exc:
            logger.error(
                "Failed to fetch annotation %s: %s",
                annotation_id,
                exc,
                exc_info=True,
            )
            raise

    def create(
        self,
        result_id: int,
        clinician_id: int | None,
        band_name: str,
        start_time_sec: float,
        end_time_sec: float | None,
        lane_start: float | None,
        lane_end: float | None,
        label: str,
        notes: str | None,
        color: str | None,
    ) -> int:
        logger.debug("Creating annotation for result %s", result_id)
        query = """
        INSERT INTO eeg_annotations(
            result_id, clinician_id, band_name, start_time_sec, end_time_sec,
            lane_start, lane_end, label, notes, color
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    query,
                    (
                        result_id,
                        clinician_id,
                        band_name,
                        start_time_sec,
                        end_time_sec,
                        lane_start,
                        lane_end,
                        label,
                        notes,
                        color,
                    ),
                )
                return cursor.fetchone()[0]
        except Exception as exc:
            logger.error(
                "Failed to create annotation for result %s: %s",
                result_id,
                exc,
                exc_info=True,
            )
            raise

    def update(
        self,
        annotation_id: int,
        start_time_sec: float,
        end_time_sec: float | None,
        lane_start: float | None,
        lane_end: float | None,
        label: str,
        notes: str | None,
        color: str | None,
    ) -> dict | None:
        logger.debug("Updating annotation %s", annotation_id)
        query = """
        UPDATE eeg_annotations
        SET start_time_sec = %s,
            end_time_sec = %s,
            lane_start = %s,
            lane_end = %s,
            label = %s,
            notes = %s,
            color = %s,
            updated_at = NOW()
        WHERE id = %s
        RETURNING id,
                  result_id,
                  clinician_id,
                  band_name,
                  start_time_sec,
                  end_time_sec,
                  lane_start,
                  lane_end,
                  label,
                  notes,
                  color,
                  created_at,
                  updated_at,
                  (
                      SELECT TRIM(CONCAT_WS(' ', c.first_name, c.last_name))
                      FROM clinicians c
                      WHERE c.id = eeg_annotations.clinician_id
                  ) AS clinician_name;
        """
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(
                    query,
                    (
                        start_time_sec,
                        end_time_sec,
                        lane_start,
                        lane_end,
                        label,
                        notes,
                        color,
                        annotation_id,
                    ),
                )
                return cursor.fetchone()
        except Exception as exc:
            logger.error(
                "Failed to update annotation %s: %s",
                annotation_id,
                exc,
                exc_info=True,
            )
            raise

    def delete(self, annotation_id: int) -> bool:
        logger.debug("Deleting annotation %s", annotation_id)
        query = "DELETE FROM eeg_annotations WHERE id = %s RETURNING id;"
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (annotation_id,))
                return cursor.fetchone() is not None
        except Exception as exc:
            logger.error(
                "Failed to delete annotation %s: %s",
                annotation_id,
                exc,
                exc_info=True,
            )
            raise
