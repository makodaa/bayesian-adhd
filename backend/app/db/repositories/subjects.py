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
    
    def get_by_subject_code(self, subject_code):
        """Get subject by subject_code."""
        logger.debug(f"Fetching subject by code: {subject_code}")
        query = "SELECT * FROM subjects WHERE subject_code = %s;"
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (subject_code,))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Subject found with code: {subject_code}")
                else:
                    logger.debug(f"Subject not found with code: {subject_code}")
                return result
        except Exception as e:
            logger.error(f"Failed to fetch subject by code {subject_code}: {e}", exc_info=True)
            raise

    def get_with_assessments(self, subject_id):
        """Get subject with their assessment results."""
        logger.debug(f"Fetching subject {subject_id} with assessments")
        query = """
        SELECT 
            s.id,
            s.subject_code,
            s.age,
            s.gender,
            s.created_at,
            r.id as result_id,
            r.predicted_class,
            r.confidence_score,
            r.inferenced_at,
            rec.file_name,
            c.first_name,
            c.last_name
        FROM subjects s
        LEFT JOIN recordings rec ON s.id = rec.subject_id
        LEFT JOIN results r ON rec.id = r.recording_id
        LEFT JOIN clinicians c ON r.clinician_id = c.id
        WHERE s.id = %s
        ORDER BY r.inferenced_at DESC;
        """
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (subject_id,))
                rows = cursor.fetchall()
                
                if not rows:
                    return None
                
                subject_data = rows[0]
                subject = {
                    'id': subject_data['id'],
                    'subject_code': subject_data['subject_code'],
                    'age': subject_data['age'],
                    'gender': subject_data['gender'],
                    'created_at': subject_data['created_at'],
                    'assessments': []
                }
                
                for row in rows:
                    if row['result_id']:
                        clinician_name = None
                        if row['first_name'] or row['last_name']:
                            clinician_name = f"{row['first_name'] or ''} {row['last_name'] or ''}".strip()
                        
                        subject['assessments'].append({
                            'result_id': row['result_id'],
                            'predicted_class': row['predicted_class'],
                            'confidence_score': row['confidence_score'],
                            'inferenced_at': row['inferenced_at'],
                            'file_name': row['file_name'],
                            'clinician_name': clinician_name
                        })
                
                return subject
        except Exception as e:
            logger.error(f"Failed to fetch subject with assessments: {e}", exc_info=True)
            raise
    
    def get_all(self):
        """Get all subjects with their details."""
        logger.debug("Fetching all subjects")
        query = """
        SELECT 
            s.*,
            MAX(r.inferenced_at) as last_update,
            (array_agg(r.predicted_class ORDER BY r.inferenced_at DESC))[1] as last_result
        FROM subjects s
        LEFT JOIN recordings rec ON s.id = rec.subject_id
        LEFT JOIN results r ON rec.id = r.recording_id
        GROUP BY s.id
        ORDER BY s.id;
        """
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
