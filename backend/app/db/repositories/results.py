from .base import BaseRepository

class ResultsRepository(BaseRepository):
    def create_result(self, recording_id, predicted_class, confidence_score):
        """Create a new result and return its ID."""
        query = """
        INSERT INTO results(recording_id, predicted_class, confidence_score)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (recording_id, predicted_class, confidence_score))
            return cursor.fetchone()[0]
    
    def get_by_id(self, result_id):
        """Get result by ID."""
        query = "SELECT * FROM results WHERE id = %s;"
        with self.get_connection() as conn:
            cursor = self.get_dict_cursor(conn)
            cursor.execute(query, (result_id,))
            return cursor.fetchone()
    
    def get_by_recording(self, recording_id):
        """Get all results for a recording."""
        query = "SELECT * FROM results WHERE recording_id = %s ORDER BY created_at DESC;"
        with self.get_connection() as conn:
            cursor = self.get_dict_cursor(conn)
            cursor.execute(query, (recording_id,))
            return cursor.fetchall()
