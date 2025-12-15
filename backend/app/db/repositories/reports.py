from .base import BaseRepository

class ReportsRepository(BaseRepository):
    def create_report(self, result_id, interpretation, report_path, clinician_id):
        query = """
        INSERT INTO reports(result_id, interpretation, report_path, clinician_id)
        VALUES (%s,%s,%s,%s)
        RETURNING id;
        """
        self.cursor.execute(query, (result_id, interpretation, report_path, clinician_id))
        return self.cursor.fetchone()[0]