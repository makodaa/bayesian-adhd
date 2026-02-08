from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)

class ResultsRepository(BaseRepository):
    def create_result(self, recording_id, classification, confidence_score, clinician_id=None):
        """Create a new result and return its ID."""
        logger.info(f"Creating result for recording {recording_id}: classification={classification}, confidence={confidence_score*100:.2f}%")
        query = """
        INSERT INTO results(recording_id, clinician_id, predicted_class, confidence_score)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (recording_id, clinician_id, classification, confidence_score))
                result_id = cursor.fetchone()[0]
                logger.info(f"Result created successfully with ID: {result_id}")
                return result_id
        except Exception as e:
            logger.error(f"Failed to create result for recording {recording_id}: {e}", exc_info=True)
            raise
    
    def get_by_id(self, result_id):
        """Get result by ID."""
        logger.debug(f"Fetching result by ID: {result_id}")
        query = "SELECT * FROM results WHERE id = %s;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (result_id,))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Result found: {result_id}")
                else:
                    logger.warning(f"Result not found: {result_id}")
                return result
        except Exception as e:
            logger.error(f"Failed to fetch result {result_id}: {e}", exc_info=True)
            raise
    
    def get_by_recording(self, recording_id):
        """Get all results for a recording."""
        logger.debug(f"Fetching all results for recording: {recording_id}")
        query = "SELECT * FROM results WHERE recording_id = %s ORDER BY inferenced_at DESC;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (recording_id,))
                results = cursor.fetchall()
                logger.debug(f"Found {len(results)} results for recording {recording_id}")
                return results
        except Exception as e:
            logger.error(f"Failed to fetch results for recording {recording_id}: {e}", exc_info=True)
            raise

    def get_latest_by_recording_id(self, recording_id):
        """Get the most recent result for a recording."""
        logger.debug(f"Fetching latest result for recording: {recording_id}")
        query = "SELECT * FROM results WHERE recording_id = %s ORDER BY inferenced_at DESC LIMIT 1;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (recording_id,))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Latest result found for recording {recording_id}: result_id={result['id']}")
                else:
                    logger.warning(f"No result found for recording {recording_id}")
                return result
        except Exception as e:
            logger.error(f"Failed to fetch latest result for recording {recording_id}: {e}", exc_info=True)
            raise
