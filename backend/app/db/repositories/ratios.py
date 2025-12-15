from .base import BaseRepository

class RatiosRepository(BaseRepository):
    def create_ratio(self, results_id, ratio_name, ratio_value):
        query = """
        INSERT INTO recordings(results_id, ratio_name, ratio_value)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        self.cursor.execute(query, (results_id, ratio_name, ratio_value))
        return self.cursor.fetchone()[0]