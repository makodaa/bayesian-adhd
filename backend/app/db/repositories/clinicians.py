from .base import BaseRepository

class CliniciansRepository(BaseRepository):
    def create_clinician(self, first_name, last_name, middle_name, occupation):
        query = """
        INSERT INTO clinicians (first_name, last_name, middle_name, occupation)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        self.cursor.execute(query, (first_name, last_name, middle_name,occupation))
        return self.cursor.fetchone()[0]