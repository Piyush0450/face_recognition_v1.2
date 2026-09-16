"""
FaceRecognition - Database Helpers Module (Stage 9 & 11)

Provides SQLite database operations for managing people, face embeddings,
and daily attendance records in database/attendance.db.
"""

from datetime import datetime
import os
import sqlite3
import numpy as np

# Determine database path relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "attendance.db")


def get_connection() -> sqlite3.Connection:
    """Create and return a database connection with Foreign Keys enabled."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Initialize SQLite database tables if they do not already exist."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Create people table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            );
        """)

        # 2. Create embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                person_id INTEGER PRIMARY KEY,
                vector BLOB NOT NULL,
                FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
            );
        """)

        # 3. Create attendance table (UNIQUE constraint ensures 1 log per person per day)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE,
                UNIQUE(person_id, date)
            );
        """)

        conn.commit()
    print("Database tables initialized successfully.")


def add_person(name: str, embedding: np.ndarray) -> bool:
    """
    Insert a person and their 512-dim embedding into SQLite database.
    Returns True if successfully added/updated, False otherwise.
    """
    name_clean = name.strip()
    if not name_clean or embedding is None:
        return False

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Convert numpy embedding vector to raw float32 BLOB bytes
    vector_bytes = embedding.astype(np.float32).tobytes()

    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            # Insert person name into people table
            cursor.execute(
                "INSERT INTO people (name, created_at) VALUES (?, ?);",
                (name_clean, created_at),
            )
            person_id = cursor.lastrowid

            # Insert embedding BLOB
            cursor.execute(
                "INSERT INTO embeddings (person_id, vector) VALUES (?, ?);",
                (person_id, sqlite3.Binary(vector_bytes)),
            )
            conn.commit()
            return True

        except sqlite3.IntegrityError:
            # Handle duplicate person name gracefully by updating existing embedding
            cursor.execute("SELECT id FROM people WHERE name = ?;", (name_clean,))
            row = cursor.fetchone()
            if row:
                person_id = row[0]
                cursor.execute(
                    "INSERT OR REPLACE INTO embeddings (person_id, vector) VALUES (?, ?);",
                    (person_id, sqlite3.Binary(vector_bytes)),
                )
                conn.commit()
                print(f"Updated existing record for '{name_clean}'.")
                return True
            return False


def get_all_people() -> list[tuple[int, str]]:
    """Return a list of (id, name) tuples for all registered people."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM people ORDER BY name ASC;")
        return cursor.fetchall()


def get_all_embeddings() -> dict[str, np.ndarray]:
    """
    Retrieve all person embeddings from SQLite database.
    Returns a dictionary mapping person name -> numpy float32 embedding array.
    """
    embeddings_dict = {}
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.name, e.vector
            FROM people p
            JOIN embeddings e ON p.id = e.person_id;
        """)
        rows = cursor.fetchall()

        for name, vector_blob in rows:
            # Reconstruct float32 numpy vector from binary BLOB
            vector_array = np.frombuffer(vector_blob, dtype=np.float32)
            embeddings_dict[name] = vector_array

    return embeddings_dict


def log_attendance(name: str) -> bool:
    """
    Record today's attendance for a person if not already logged today.
    Returns True if attendance was logged, False if already logged today or person not found.
    """
    today_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM people WHERE name = ?;", (name.strip(),))
        row = cursor.fetchone()

        if not row:
            return False

        person_id = row[0]

        try:
            # Insert attendance (UNIQUE(person_id, date) prevents duplicate daily logs)
            cursor.execute(
                "INSERT INTO attendance (person_id, date, time) VALUES (?, ?, ?);",
                (person_id, today_date, current_time),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Attendance already logged for today
            return False


def get_today_attendance() -> list[tuple[str, str]]:
    """Return a list of (name, time) tuples for today's attendance logs."""
    return get_attendance_by_date(datetime.now().strftime("%Y-%m-%d"))


def get_attendance_by_date(date_str: str) -> list[tuple[str, str]]:
    """Return a list of (name, time) tuples for attendance logged on the given date (YYYY-MM-DD)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.name, a.time
            FROM attendance a
            JOIN people p ON a.person_id = p.id
            WHERE a.date = ?
            ORDER BY a.time ASC;
        """, (date_str,))
        return cursor.fetchall()


def delete_person_by_name(name: str) -> bool:
    """Delete a person record from database (cascades to embeddings and attendance)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM people WHERE name = ?;", (name.strip(),))
        conn.commit()
        return cursor.rowcount > 0
