import sqlite3
import os

DB_NAME = "patients.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age TEXT,
            gender TEXT,
            volume REAL,
            risk TEXT,
            confidence REAL,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_patient(patient_data):
    """
    patient_data tuple structure:
    (patient_id, name, age, gender, volume_val, risk, confidence, date_str)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patients (patient_id, name, age, gender, volume, risk, confidence, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, patient_data)
    conn.commit()
    conn.close()

def get_all_patients():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [tuple(row) for row in rows]

def get_patient(patient_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return tuple(row) if row else None

def get_patient_history(name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE name = ? ORDER BY id ASC", (name,))
    rows = cursor.fetchall()
    conn.close()
    return [tuple(row) for row in rows]