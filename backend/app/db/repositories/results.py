from .base import BaseRepository

class ResultsRepository(BaseRepository):
    def create_result(self, recording_id, predicted_class, confidence_score):
        query = """
        INSERT INTO results(recording_id, predicted_class, confidence_score)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        self.cursor.execute(query, (recording_id, predicted_class, confidence_score))
        return self.cursor.fetchone()[0]
