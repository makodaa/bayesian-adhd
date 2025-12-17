from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)

class CliniciansRepository(BaseRepository):
    def create_clinician(self, first_name, last_name, middle_name, occupation):
        """Create a new clinician and return its ID."""
        logger.info(f"Creating clinician: {first_name} {last_name}, occupation={occupation}")
        query = """
        INSERT INTO clinicians (first_name, last_name, middle_name, occupation)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (first_name, last_name, middle_name, occupation))
                clinician_id = cursor.fetchone()[0]
                logger.info(f"Clinician created successfully with ID: {clinician_id}")
                return clinician_id
        except Exception as e:
            logger.error(f"Failed to create clinician: {e}", exc_info=True)
            raise
    
    def get_by_id(self, clinician_id):
        """Get clinician by ID."""
        logger.debug(f"Fetching clinician by ID: {clinician_id}")
        query = "SELECT * FROM clinicians WHERE id = %s;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (clinician_id,))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Clinician found: {clinician_id}")
                else:
                    logger.warning(f"Clinician not found: {clinician_id}")
                return result
        except Exception as e:
            logger.error(f"Failed to fetch clinician {clinician_id}: {e}", exc_info=True)
            raise
    
    def get_by_name(self, first_name, last_name):
        """Get clinician by first and last name."""
        logger.debug(f"Fetching clinician by name: {first_name} {last_name}")
        query = "SELECT * FROM clinicians WHERE first_name = %s AND last_name = %s;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (first_name, last_name))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Clinician found: {first_name} {last_name}")
                else:
                    logger.debug(f"Clinician not found: {first_name} {last_name}")
                return result
        except Exception as e:
            logger.error(f"Failed to fetch clinician by name: {e}", exc_info=True)
            raise
    
    def get_all(self):
        """Get all clinicians."""
        logger.debug("Fetching all clinicians")
        query = "SELECT * FROM clinicians ORDER BY id;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query)
                results = cursor.fetchall()
                logger.info(f"Retrieved {len(results)} clinicians")
                return results
        except Exception as e:
            logger.error(f"Failed to fetch all clinicians: {e}", exc_info=True)
            raise