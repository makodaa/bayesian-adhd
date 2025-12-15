from .base import BaseRepository

class ResultsRepository(BaseRepository):
    def create_result(self, recording_id, predicted_class, confidence_score, inferenced_at):
        query = """
        INSERT INTO results(recording_id, predicted_class, confidence_score, inferenced_at)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        self.cursor.execute(query, (recording_id, predicted_class, confidence_score, inferenced_at))
        return self.cursor.fetchone()[0]
