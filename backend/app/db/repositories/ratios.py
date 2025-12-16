from .base import BaseRepository

class RatiosRepository(BaseRepository):
    def create_ratio(self, result_id, ratio_name, ratio_value):
        """Create a new ratio entry and return its ID."""
        query = """
        INSERT INTO ratios(result_id, ratio_name, ratio_value)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (result_id, ratio_name, ratio_value))
            return cursor.fetchone()[0]
    
    def get_by_result(self, result_id):
        """Get all ratios for a result."""
        query = "SELECT * FROM ratios WHERE result_id = %s ORDER BY ratio_name;"
        with self.get_connection() as conn:
            cursor = self.get_dict_cursor(conn)
            cursor.execute(query, (result_id,))
            return cursor.fetchall()