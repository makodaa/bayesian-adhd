import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import os
from ..core.logging_config import get_db_logger

logger = get_db_logger(__name__)

DB_CONFIG = {
    "host": os.getenv("DATABASE_HOST", "localhost"),
    "port": int(os.getenv("DATABASE_PORT", 5432)),
    "dbname": os.getenv("DATABASE_NAME", "bayesian_adhd"),
    "user": os.getenv("DATABASE_USER", "db_user"),
    "password": os.getenv("DATABASE_PASSWORD", "db_password"),
}

@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Automatically handles commits, rollbacks, and closing connections.
    
    Usage:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
            results = cursor.fetchall()
    """
    conn = None
    try:
        logger.debug(f"Attempting database connection to {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
        conn = psycopg2.connect(**DB_CONFIG)
        logger.debug("Database connection established successfully")
        yield conn
        conn.commit()
        logger.debug("Database transaction committed successfully")
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}", exc_info=True)
        if conn:
            conn.rollback()
            logger.warning("Database transaction rolled back")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in database connection: {e}", exc_info=True)
        if conn:
            conn.rollback()
            logger.warning("Database transaction rolled back")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed")

def get_dict_cursor(conn):
    """Get a cursor that returns results as dictionaries."""
    return conn.cursor(cursor_factory=RealDictCursor)