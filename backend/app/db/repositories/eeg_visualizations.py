from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)


class EEGVisualizationsRepository(BaseRepository):
    """Repository for persisted EEG visualization images (base64 PNG)."""

    def create_visualization(
        self, result_id: int, band_name: str, image_data: str
    ) -> int:
        """Save an EEG visualization image for a result.

        Uses INSERT ... ON CONFLICT to upsert so re-runs don't fail.
        """
        logger.debug(
            f"Saving EEG visualization for result {result_id}: band={band_name}"
        )
        query = """
        INSERT INTO eeg_visualizations(result_id, band_name, image_data)
        VALUES (%s, %s, %s)
        ON CONFLICT (result_id, band_name) DO UPDATE
            SET image_data = EXCLUDED.image_data,
                created_at = NOW()
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (result_id, band_name, image_data))
                viz_id = cursor.fetchone()[0]
                logger.debug(f"EEG visualization created/updated with ID: {viz_id}")
                return viz_id
        except Exception as e:
            logger.error(
                f"Failed to save EEG visualization for result {result_id}: {e}",
                exc_info=True,
            )
            raise

    def get_by_result(self, result_id: int) -> list[dict]:
        """Get all EEG visualization images for a result."""
        logger.debug(f"Fetching EEG visualizations for result: {result_id}")
        query = """
        SELECT id, result_id, band_name, image_data, created_at
        FROM eeg_visualizations
        WHERE result_id = %s
        ORDER BY id;
        """
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (result_id,))
                results = cursor.fetchall()
                logger.info(
                    f"Retrieved {len(results)} EEG visualizations for result {result_id}"
                )
                return results
        except Exception as e:
            logger.error(
                f"Failed to fetch EEG visualizations for result {result_id}: {e}",
                exc_info=True,
            )
            raise

    def get_by_result_and_band(
        self, result_id: int, band_name: str
    ) -> dict | None:
        """Get a single EEG visualization by result and band name."""
        logger.debug(
            f"Fetching EEG visualization for result {result_id}, band={band_name}"
        )
        query = """
        SELECT id, result_id, band_name, image_data, created_at
        FROM eeg_visualizations
        WHERE result_id = %s AND band_name = %s;
        """
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (result_id, band_name))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"EEG visualization found: {band_name}")
                else:
                    logger.debug(f"No EEG visualization for band: {band_name}")
                return result
        except Exception as e:
            logger.error(
                f"Failed to fetch EEG visualization for result {result_id}, "
                f"band={band_name}: {e}",
                exc_info=True,
            )
            raise
