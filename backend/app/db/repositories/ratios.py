from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)

class RatiosRepository(BaseRepository):
    def create_ratio(self, result_id, ratio_name, ratio_value):
        """Create a new ratio entry and return its ID."""
        logger.debug(f"Creating ratio for result {result_id}: {ratio_name}={ratio_value:.4f}")

        query = """
        INSERT INTO ratios(result_id, ratio_name, ratio_value)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (result_id, ratio_name, ratio_value))
                ratio_id = cursor.fetchone()[0]
                logger.debug(f"Ratio created with ID: {ratio_id}")
                return ratio_id
        except Exception as e:
            logger.error(f"Failed to create ratio for result {result_id}: {e}", exc_info=True)
            raise

    def get_by_result(self, result_id):
        """Get all ratios for a result."""
        logger.debug(f"Fetching ratios for result: {result_id}")
        query = "SELECT * FROM ratios WHERE result_id = %s ORDER BY ratio_name;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (result_id,))
                results = cursor.fetchall()
                logger.info(f"Retrieved {len(results)} ratios for result {result_id}")
                return results
        except Exception as e:
            logger.error(f"Failed to fetch ratios for result {result_id}: {e}", exc_info=True)
            raise