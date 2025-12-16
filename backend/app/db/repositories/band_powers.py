from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)

class BandPowersRepository(BaseRepository):
    def create_band_power(self, result_id, electrode, frequency_band, absolute_power, relative_power):
        """Create a new band power entry and return its ID."""
        logger.debug(f"Creating band power for result {result_id}: electrode={electrode}, band={frequency_band}")
        query = """
        INSERT INTO band_powers(result_id, electrode, frequency_band, absolute_power, relative_power)
        VALUES (%s,%s,%s,%s,%s)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (result_id, electrode, frequency_band, absolute_power, relative_power))
                band_power_id = cursor.fetchone()[0]
                logger.debug(f"Band power created with ID: {band_power_id}")
                return band_power_id
        except Exception as e:
            logger.error(f"Failed to create band power for result {result_id}: {e}", exc_info=True)
            raise
    
    def get_by_result(self, result_id):
        """Get all band powers for a result."""
        logger.debug(f"Fetching band powers for result: {result_id}")
        query = "SELECT * FROM band_powers WHERE result_id = %s ORDER BY electrode, frequency_band;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (result_id,))
                results = cursor.fetchall()
                logger.info(f"Retrieved {len(results)} band powers for result {result_id}")
                return results
        except Exception as e:
            logger.error(f"Failed to fetch band powers for result {result_id}: {e}", exc_info=True)
            raise
    