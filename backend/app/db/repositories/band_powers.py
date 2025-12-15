from .base import BaseRepository

class BandPowersRepository(BaseRepository):
    def create_band_power(self, result_id, frequency_band, absolute_power, relative_power):
        query = """
        INSERT INTO band_powers(result_id, frequency_band, absolute_power, relative_power)
        VALUES (%s,%s,%s,%s)
        RETURNING id;
        """
        self.cursor.execute(query, (result_id, frequency_band, absolute_power, relative_power))
        return self.cursor.fetchone()[0]
    