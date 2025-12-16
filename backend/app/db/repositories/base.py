from ..connection import get_db_connection, get_dict_cursor

class BaseRepository:
    """Base repository with database connection utilities."""
    
    @staticmethod
    def get_connection():
        """Get database connection context manager."""
        return get_db_connection()
    
    @staticmethod
    def get_dict_cursor(conn):
        """Get cursor that returns dictionary results."""
        return get_dict_cursor(conn)