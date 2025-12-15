from .base import BaseRepository

class RecordingsRepository(BaseRepository):
    def create_recording(self, subject_id, file_name, file_path):
        query = """
        INSERT INTO recordings(subject_id, file_name, file_path)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        self.cursor.execute(query, (subject_id, file_name, file_path))
        return self.cursor.fetchone()[0]