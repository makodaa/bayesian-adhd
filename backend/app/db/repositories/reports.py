from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)

class ReportsRepository(BaseRepository):
    def create_report(self, result_id, interpretation, report_path, clinician_id):
        """Create a new report and return its ID."""
        logger.info(f"Creating report for result {result_id} by clinician {clinician_id}")
        query = """
        INSERT INTO reports(result_id, interpretation, report_path, clinician_id)
        VALUES (%s,%s,%s,%s)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (result_id, interpretation, report_path, clinician_id))
                report_id = cursor.fetchone()[0]
                logger.info(f"Report created successfully with ID: {report_id}")
                return report_id
        except Exception as e:
            logger.error(f"Failed to create report for result {result_id}: {e}", exc_info=True)
            raise
    
    def get_by_id(self, report_id):
        """Get report by ID."""
        logger.debug(f"Fetching report by ID: {report_id}")
        query = "SELECT * FROM reports WHERE id = %s;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (report_id,))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Report found: {report_id}")
                else:
                    logger.warning(f"Report not found: {report_id}")
                return result
        except Exception as e:
            logger.error(f"Failed to fetch report {report_id}: {e}", exc_info=True)
            raise
    
    def get_by_result(self, result_id):
        """Get all reports for a result."""
        logger.debug(f"Fetching all reports for result: {result_id}")
        query = "SELECT * FROM reports WHERE result_id = %s ORDER BY created_at DESC;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (result_id,))
                results = cursor.fetchall()
                logger.info(f"Retrieved {len(results)} reports for result {result_id}")
                return results
        except Exception as e:
            logger.error(f"Failed to fetch reports for result {result_id}: {e}", exc_info=True)
            raise