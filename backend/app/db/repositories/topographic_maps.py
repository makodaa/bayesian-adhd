from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)


class TopographicMapsRepository(BaseRepository):
    def create_map(self, result_id, map_type, band, map_image):
        """Save a topographic map image for a result."""
        logger.debug(f"Saving topographic map for result {result_id}: type={map_type}, band={band}")
        query = """
        INSERT INTO topographic_maps(result_id, map_type, band, map_image)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (result_id, map_type, band, map_image))
                map_id = cursor.fetchone()[0]
                logger.debug(f"Topographic map created with ID: {map_id}")
                return map_id
        except Exception as e:
            logger.error(f"Failed to save topographic map for result {result_id}: {e}", exc_info=True)
            raise

    def get_by_result(self, result_id):
        """Get all topographic maps for a result."""
        logger.debug(f"Fetching topographic maps for result: {result_id}")
        query = "SELECT * FROM topographic_maps WHERE result_id = %s ORDER BY map_type, band;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (result_id,))
                results = cursor.fetchall()
                logger.info(f"Retrieved {len(results)} topographic maps for result {result_id}")
                return results
        except Exception as e:
            logger.error(f"Failed to fetch topographic maps for result {result_id}: {e}", exc_info=True)
            raise
