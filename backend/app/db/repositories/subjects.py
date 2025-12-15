from .base import BaseRepository

class SubjectsRepository(BaseRepository):
    def create_subject(self, subject_code, age, sex):
        query = """
        INSERT INTO subjects(subject_code, age, sex)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        self.cursor.execute(query, (subject_code, age, sex))
        return self.cursor.fetchone()[0]
