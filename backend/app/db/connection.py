import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import os

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
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def get_dict_cursor(conn):
    """Get a cursor that returns results as dictionaries."""
    return conn.cursor(cursor_factory=RealDictCursor)