from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)


class ClinicianSubtypeRecommendationsRepository(BaseRepository):
    def get_by_clinician(self, clinician_id: int) -> list[dict]:
        query = """
        SELECT clinician_id, adhd_subtype, trigger_key, recommendation_text
        FROM clinician_subtype_recommendations
        WHERE clinician_id = %s
        ORDER BY adhd_subtype, trigger_key;
        """
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (clinician_id,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(
                "Failed to fetch clinician subtype recommendations: %s",
                e,
                exc_info=True,
            )
            raise

    def replace_for_clinician(self, clinician_id: int, rows: list[dict]) -> None:
        delete_query = (
            "DELETE FROM clinician_subtype_recommendations WHERE clinician_id = %s;"
        )
        insert_query = """
        INSERT INTO clinician_subtype_recommendations
            (clinician_id, adhd_subtype, trigger_key, recommendation_text, created_at, updated_at)
        VALUES
            (%s, %s, %s, %s, NOW(), NOW());
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(delete_query, (clinician_id,))
                if rows:
                    cursor.executemany(
                        insert_query,
                        [
                            (
                                clinician_id,
                                row["adhd_subtype"],
                                row["trigger_key"],
                                row["recommendation_text"],
                            )
                            for row in rows
                        ],
                    )
        except Exception as e:
            logger.error(
                "Failed to replace clinician subtype recommendations: %s",
                e,
                exc_info=True,
            )
            raise
