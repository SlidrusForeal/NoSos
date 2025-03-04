import sqlite3
import logging

def get_connection(db_filename: str):
    try:
        conn = sqlite3.connect(db_filename, check_same_thread=False)
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise

def create_tables(conn):
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity (
                player TEXT,
                time REAL,
                hour INTEGER,
                date TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS zones (
                player TEXT,
                zone TEXT,
                time REAL,
                date TEXT
            )
        ''')
        conn.commit()
    except Exception as e:
        logging.error(f"Error creating tables: {e}")
