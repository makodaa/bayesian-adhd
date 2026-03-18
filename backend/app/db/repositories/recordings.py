from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)

class RecordingsRepository(BaseRepository):
    def create_recording(
        self,
        subject_id,
        file_name,
        referral_name=None,
        referral_institution=None,
        technician_name=None,
        sleep_hours=None,
        coffee_hours_ago=None,
        drugs_hours_ago=None,
        meal_hours_ago=None,
        medication=None,
        recorded_minutes=None,
        duration_minutes=None,
        artifacts_noted=None,
        notes=None,
    ):
        """Create a new recording and return its ID."""
        logger.info(f"Creating recording for subject {subject_id}: file={file_name}")
        query = """
        INSERT INTO recordings(
            subject_id,
            file_name,
            referral_name,
            referral_institution,
            technician_name,
            sleep_hours,
            coffee_hours_ago,
            drugs_hours_ago,
            meal_hours_ago,
            medication,
            recorded_minutes,
            duration_minutes,
            artifacts_noted,
            notes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    query,
                    (
                        subject_id,
                        file_name,
                        referral_name,
                        referral_institution,
                        technician_name,
                        sleep_hours,
                        coffee_hours_ago,
                        drugs_hours_ago,
                        meal_hours_ago,
                        medication,
                        recorded_minutes,
                        duration_minutes,
                        artifacts_noted,
                        notes,
                    ),
                )
                recording_id = cursor.fetchone()[0]
                logger.info(f"Recording created successfully with ID: {recording_id}")
                return recording_id
        except Exception as e:
            logger.error(f"Failed to create recording for subject {subject_id}: {e}", exc_info=True)
            raise

    def get_by_id(self, recording_id):
        """Get recording by ID."""
        logger.debug(f"Fetching recording by ID: {recording_id}")
        query = "SELECT * FROM recordings WHERE id = %s;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (recording_id,))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Recording found: {recording_id}")
                else:
                    logger.warning(f"Recording not found: {recording_id}")
                return result
        except Exception as e:
            logger.error(f"Failed to fetch recording {recording_id}: {e}", exc_info=True)
            raise

    def get_by_subject(self, subject_id):
        """Get all recordings for a subject."""
        logger.debug(f"Fetching all recordings for subject: {subject_id}")
        query = "SELECT * FROM recordings WHERE subject_id = %s ORDER BY uploaded_at DESC;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (subject_id,))
                results = cursor.fetchall()
                logger.info(f"Retrieved {len(results)} recordings for subject {subject_id}")
                return results
        except Exception as e:
            logger.error(f"Failed to fetch recordings for subject {subject_id}: {e}", exc_info=True)
            raise
