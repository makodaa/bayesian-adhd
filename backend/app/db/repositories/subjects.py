from .base import BaseRepository
from ...core.logging_config import get_db_logger

logger = get_db_logger(__name__)

class SubjectsRepository(BaseRepository):
    def create_subject(self, subject_code, age, sex):
        """Create a new subject and return its ID."""
        logger.info(f"Creating subject: code={subject_code}, age={age}, sex={sex}")
        query = """
        INSERT INTO subjects(subject_code, age, gender)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (subject_code, age, sex))
                subject_id = cursor.fetchone()[0]
                logger.info(f"Subject created successfully with ID: {subject_id}")
                return subject_id
        except Exception as e:
            logger.error(f"Failed to create subject: {e}", exc_info=True)
            raise
    
    def get_by_id(self, subject_id):
        """Get subject by ID."""
        logger.debug(f"Fetching subject by ID: {subject_id}")
        query = "SELECT * FROM subjects WHERE id = %s;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (subject_id,))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Subject found: {subject_id}")
                else:
                    logger.warning(f"Subject not found: {subject_id}")
                return result
        except Exception as e:
            logger.error(f"Failed to fetch subject {subject_id}: {e}", exc_info=True)
            raise
    
    def get_all(self):
        """Get all subjects."""
        logger.debug("Fetching all subjects")
        query = "SELECT * FROM subjects ORDER BY id;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query)
                results = cursor.fetchall()
                logger.info(f"Retrieved {len(results)} subjects")
                return results
        except Exception as e:
            logger.error(f"Failed to fetch all subjects: {e}", exc_info=True)
            raise
