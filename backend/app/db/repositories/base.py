from ..connection import get_db_connection, get_dict_cursor

class BaseRepository:
    """Base repository with database connection utilities."""
    
    def get_connection(self):
        """Get database connection context manager."""
        return get_db_connection()
    
    def get_dict_cursor(self, conn):
        """Get cursor that returns dictionary results."""
        return get_dict_cursor(conn)