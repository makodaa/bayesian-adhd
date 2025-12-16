from .base import BaseRepository

class RecordingsRepository(BaseRepository):
    def create_recording(self, subject_id, file_name):
        """Create a new recording and return its ID."""
        query = """
        INSERT INTO recordings(subject_id, file_name)
        VALUES (%s, %s)
        RETURNING id;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (subject_id, file_name))
            return cursor.fetchone()[0]
    
    def get_by_id(self, recording_id):
        """Get recording by ID."""
        query = "SELECT * FROM recordings WHERE id = %s;"
        with self.get_connection() as conn:
            cursor = self.get_dict_cursor(conn)
            cursor.execute(query, (recording_id,))
            return cursor.fetchone()
    
    def get_by_subject(self, subject_id):
        """Get all recordings for a subject."""
        query = "SELECT * FROM recordings WHERE subject_id = %s ORDER BY created_at DESC;"
        with self.get_connection() as conn:
            cursor = self.get_dict_cursor(conn)
            cursor.execute(query, (subject_id,))
            return cursor.fetchall()