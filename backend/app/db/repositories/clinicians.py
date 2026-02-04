from .base import BaseRepository
from ...core.logging_config import get_db_logger
from werkzeug.security import generate_password_hash, check_password_hash

logger = get_db_logger(__name__)

class CliniciansRepository(BaseRepository):
    def create_clinician(self, first_name, last_name, middle_name, occupation, password=None):
        """Create a new clinician and return its ID."""
        logger.info(f"Creating clinician: {first_name} {last_name}, occupation={occupation}")
        password_hash = generate_password_hash(password) if password else None
        query = """
        INSERT INTO clinicians (first_name, last_name, middle_name, occupation, password_hash)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (first_name, last_name, middle_name, occupation, password_hash))
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

    def get_with_assessments(self, clinician_id):
        """Get clinician with their assessment results."""
        logger.debug(f"Fetching clinician {clinician_id} with assessments")
        query = """
        SELECT 
            c.id,
            c.first_name,
            c.last_name,
            c.middle_name,
            c.occupation,
            r.id as result_id,
            r.predicted_class,
            r.confidence_score,
            r.inferenced_at,
            s.subject_code,
            s.age,
            s.gender
        FROM clinicians c
        LEFT JOIN results r ON c.id = r.clinician_id
        LEFT JOIN recordings rec ON r.recording_id = rec.id
        LEFT JOIN subjects s ON rec.subject_id = s.id
        WHERE c.id = %s
        ORDER BY r.inferenced_at DESC;
        """
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (clinician_id,))
                rows = cursor.fetchall()
                
                if not rows:
                    return None
                
                clinician_data = rows[0]
                clinician = {
                    'id': clinician_data['id'],
                    'first_name': clinician_data['first_name'],
                    'last_name': clinician_data['last_name'],
                    'middle_name': clinician_data['middle_name'],
                    'occupation': clinician_data['occupation'],
                    'assessments': []
                }
                
                for row in rows:
                    if row['result_id']:
                        clinician['assessments'].append({
                            'result_id': row['result_id'],
                            'predicted_class': row['predicted_class'],
                            'confidence_score': row['confidence_score'],
                            'inferenced_at': row['inferenced_at'],
                            'subject_code': row['subject_code'],
                            'age': row['age'],
                            'gender': row['gender']
                        })
                
                return clinician
        except Exception as e:
            logger.error(f"Failed to fetch clinician with assessments: {e}", exc_info=True)
            raise
    
    def get_all(self):
        """Get all clinicians with their assessment count and latest activity."""
        logger.debug("Fetching all clinicians")
        query = """
        SELECT 
            c.*,
            COUNT(DISTINCT r.id) as assessments_count,
            MAX(r.inferenced_at) as last_activity,
            BOOL_OR(s.clinician_id IS NOT NULL) as is_active
        FROM clinicians c
        LEFT JOIN results r ON c.id = r.clinician_id
        LEFT JOIN clinician_sessions s ON c.id = s.clinician_id
        GROUP BY c.id
        ORDER BY c.id;
        """
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

    def set_active(self, clinician_id):
        """Mark clinician as active (logged in)."""
        logger.info(f"Setting clinician active: {clinician_id}")
        query = """
        INSERT INTO clinician_sessions (clinician_id, logged_in_at)
        VALUES (%s, NOW())
        ON CONFLICT (clinician_id)
        DO UPDATE SET logged_in_at = NOW();
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (clinician_id,))
        except Exception as e:
            logger.error(f"Failed to set clinician active: {e}", exc_info=True)
            raise

    def set_inactive(self, clinician_id):
        """Mark clinician as inactive (logged out)."""
        logger.info(f"Setting clinician inactive: {clinician_id}")
        query = "DELETE FROM clinician_sessions WHERE clinician_id = %s;"
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (clinician_id,))
        except Exception as e:
            logger.error(f"Failed to set clinician inactive: {e}", exc_info=True)
            raise

    def is_active(self, clinician_id):
        """Return True if clinician has an active session row."""
        logger.debug(f"Checking active session for clinician: {clinician_id}")
        query = "SELECT 1 FROM clinician_sessions WHERE clinician_id = %s LIMIT 1;"
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (clinician_id,))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check active session: {e}", exc_info=True)
            # On error, be conservative and treat as not active to avoid accidental lockout
            return False
    
    def get_by_username(self, username):
        """Get clinician by username (first_name + last_name combined)."""
        logger.debug(f"Fetching clinician by username: {username}")
        query = """
        SELECT * FROM clinicians 
        WHERE CONCAT(first_name, ' ', last_name) = %s;
        """
        try:
            with self.get_connection() as conn:
                cursor = self.get_dict_cursor(conn)
                cursor.execute(query, (username,))
                result = cursor.fetchone()
                if result:
                    logger.debug(f"Clinician found: {username}")
                else:
                    logger.debug(f"Clinician not found: {username}")
                return result
        except Exception as e:
            logger.error(f"Failed to fetch clinician by username: {e}", exc_info=True)
            raise
    
    def verify_password(self, clinician_id, password):
        """Verify password for a clinician."""
        logger.debug(f"Verifying password for clinician: {clinician_id}")
        try:
            clinician = self.get_by_id(clinician_id)
            if not clinician or not clinician.get('password_hash'):
                logger.warning(f"Clinician {clinician_id} not found or has no password hash")
                return False
            
            is_valid = check_password_hash(clinician['password_hash'], password)
            if is_valid:
                logger.info(f"Password verified for clinician: {clinician_id}")
            else:
                logger.warning(f"Invalid password for clinician: {clinician_id}")
            return is_valid
        except Exception as e:
            logger.error(f"Failed to verify password: {e}", exc_info=True)
            return False