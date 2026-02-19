from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)


class TemporalPlotsRepository(BaseRepository):
    def create_plot(self, result_id, group_name, plot_image):
        """Save a temporal biomarker plot image for a result."""
        logger.debug(f"Saving temporal plot for result {result_id}: group={group_name}")
        query = """
        INSERT INTO temporal_plots(result_id, group_name, plot_image)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (result_id, group_name, plot_image))
                plot_id = cursor.fetchone()[0]
                logger.debug(f"Temporal plot created with ID: {plot_id}")
                return plot_id
        except Exception as e:
            logger.error(f"Failed to save temporal plot for result {result_id}: {e}", exc_info=True)
            raise

    def get_by_result(self, result_id):
        """Get all temporal plots for a result."""
        logger.debug(f"Fetching temporal plots for result: {result_id}")
        query = "SELECT * FROM temporal_plots WHERE result_id = %s ORDER BY id;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (result_id,))
                results = cursor.fetchall()
                logger.info(f"Retrieved {len(results)} temporal plots for result {result_id}")
                return results
        except Exception as e:
            logger.error(f"Failed to fetch temporal plots for result {result_id}: {e}", exc_info=True)
            raise
