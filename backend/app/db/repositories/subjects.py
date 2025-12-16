from .base import BaseRepository

class SubjectsRepository(BaseRepository):
    def create_subject(self, subject_code, age, sex):
        """Create a new subject and return its ID."""
        query = """
        INSERT INTO subjects(subject_code, age, sex)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (subject_code, age, sex))
            return cursor.fetchone()[0]
    
    def get_by_id(self, subject_id):
        """Get subject by ID."""
        query = "SELECT * FROM subjects WHERE id = %s;"
        with self.get_connection() as conn:
            cursor = self.get_dict_cursor(conn)
            cursor.execute(query, (subject_id,))
            return cursor.fetchone()
    
    def get_all(self):
        """Get all subjects."""
        query = "SELECT * FROM subjects ORDER BY id;"
        with self.get_connection() as conn:
            cursor = self.get_dict_cursor(conn)
            cursor.execute(query)
            return cursor.fetchall()
