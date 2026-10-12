import sqlite3
import os
import json
from datetime import datetime
from config import DATABASE_PATH

def get_db_connection():
    db_dir = os.path.dirname(DATABASE_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Children table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS children (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alias TEXT NOT NULL,
            age_months INTEGER NOT NULL,
            gender TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Monitoring Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS monitoring_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id INTEGER,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            input_type TEXT NOT NULL,
            risk_level TEXT DEFAULT 'LOW',
            risk_score INTEGER DEFAULT 0,
            FOREIGN KEY (child_id) REFERENCES children (id)
        )
    ''')
    
    # Audio Records table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            audio_type TEXT DEFAULT 'UNKNOWN',
            duration REAL,
            sample_rate INTEGER DEFAULT 16000,
            source TEXT DEFAULT 'UPLOAD',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES monitoring_sessions (id)
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_id INTEGER NOT NULL,
            sound_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            screening_prediction TEXT NOT NULL,
            probabilities_json TEXT,
            cry_score INTEGER DEFAULT 0,
            cough_score INTEGER DEFAULT 0,
            resp_score INTEGER DEFAULT 0,
            risk_score INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (audio_id) REFERENCES audio_records (id)
        )
    ''')
    
    # Alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            alert_level TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES monitoring_sessions (id)
        )
    ''')
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully!")
