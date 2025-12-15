import psycopg2
from psycopg.rows import dict_row

def get_connection():
    return psycopg2.connect(
        dsn=DATABASE_URL,
        row_factory=dict_row
    )