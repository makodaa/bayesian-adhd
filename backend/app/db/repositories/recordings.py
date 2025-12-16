from .base import BaseRepository

class RecordingsRepository(BaseRepository):
    def create_recording(self, subject_id, file_name, sleep_hours=None, food_intake=None,
                        caffeinated=None, medicated=None, medication_intake=None,
                        artifacts_noted=None, notes=None):
        """Create a new recording and return its ID."""
        query = """
        INSERT INTO recordings(subject_id, file_name, sleep_hours, food_intake,
                             caffeinated, medicated, medication_intake,
                             artifacts_noted, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (subject_id, file_name, sleep_hours, food_intake,
                                 caffeinated, medicated, medication_intake,
                                 artifacts_noted, notes))
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