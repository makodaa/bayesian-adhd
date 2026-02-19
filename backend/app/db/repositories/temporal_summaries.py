from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)


class TemporalSummariesRepository(BaseRepository):
    def create_summary(self, result_id, biomarker_key, mean_value, std_value, min_value, max_value):
        """Save a temporal biomarker summary statistic for a result."""
        logger.debug(f"Saving temporal summary for result {result_id}: {biomarker_key}")
        query = """
        INSERT INTO temporal_summaries(result_id, biomarker_key, mean_value, std_value, min_value, max_value)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (result_id, biomarker_key, mean_value, std_value, min_value, max_value))
                summary_id = cursor.fetchone()[0]
                logger.debug(f"Temporal summary created with ID: {summary_id}")
                return summary_id
        except Exception as e:
            logger.error(f"Failed to save temporal summary for result {result_id}: {e}", exc_info=True)
            raise

    def get_by_result(self, result_id):
        """Get all temporal summaries for a result."""
        logger.debug(f"Fetching temporal summaries for result: {result_id}")
        query = "SELECT * FROM temporal_summaries WHERE result_id = %s ORDER BY biomarker_key;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (result_id,))
                results = cursor.fetchall()
                logger.info(f"Retrieved {len(results)} temporal summaries for result {result_id}")
                return results
        except Exception as e:
            logger.error(f"Failed to fetch temporal summaries for result {result_id}: {e}", exc_info=True)
            raise
