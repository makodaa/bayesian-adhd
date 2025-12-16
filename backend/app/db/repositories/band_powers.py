from .base import BaseRepository

class BandPowersRepository(BaseRepository):
    def create_band_power(self, result_id, electrode, frequency_band, absolute_power, relative_power):
        """Create a new band power entry and return its ID."""
        query = """
        INSERT INTO band_powers(result_id, electrode, frequency_band, absolute_power, relative_power)
        VALUES (%s,%s,%s,%s,%s)
        RETURNING id;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (result_id, electrode, frequency_band, absolute_power, relative_power))
            return cursor.fetchone()[0]
    
    def get_by_result(self, result_id):
        """Get all band powers for a result."""
        query = "SELECT * FROM band_powers WHERE result_id = %s ORDER BY electrode, frequency_band;"
        with self.get_connection() as conn:
            cursor = self.get_dict_cursor(conn)
            cursor.execute(query, (result_id,))
            return cursor.fetchall()
    