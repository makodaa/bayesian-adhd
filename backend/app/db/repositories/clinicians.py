from .base import BaseRepository

class CliniciansRepository(BaseRepository):
    def create_clinician(self, first_name, last_name, middle_name, occupation):
        """Create a new clinician and return its ID."""
        query = """
        INSERT INTO clinicians (first_name, last_name, middle_name, occupation)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (first_name, last_name, middle_name, occupation))
            return cursor.fetchone()[0]
    
    def get_by_id(self, clinician_id):
        """Get clinician by ID."""
        query = "SELECT * FROM clinicians WHERE id = %s;"
        with self.get_connection() as conn:
            cursor = self.get_dict_cursor(conn)
            cursor.execute(query, (clinician_id,))
            return cursor.fetchone()
    
    def get_all(self):
        """Get all clinicians."""
        query = "SELECT * FROM clinicians ORDER BY id;"
        with self.get_connection() as conn:
            cursor = self.get_dict_cursor(conn)
            cursor.execute(query)
            return cursor.fetchall()