from .base import BaseRepository

class ReportsRepository(BaseRepository):
    def create_report(self, result_id, interpretation, report_path, clinician_id):
        """Create a new report and return its ID."""
        query = """
        INSERT INTO reports(result_id, interpretation, report_path, clinician_id)
        VALUES (%s,%s,%s,%s)
        RETURNING id;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (result_id, interpretation, report_path, clinician_id))
            return cursor.fetchone()[0]
    
    def get_by_id(self, report_id):
        """Get report by ID."""
        query = "SELECT * FROM reports WHERE id = %s;"
        with self.get_connection() as conn:
            cursor = self.get_dict_cursor(conn)
            cursor.execute(query, (report_id,))
            return cursor.fetchone()
    
    def get_by_result(self, result_id):
        """Get all reports for a result."""
        query = "SELECT * FROM reports WHERE result_id = %s ORDER BY created_at DESC;"
        with self.get_connection() as conn:
            cursor = self.get_dict_cursor(conn)
            cursor.execute(query, (result_id,))
            return cursor.fetchall()