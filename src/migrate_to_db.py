"""
FaceRecognition - Database Migration Script (Stage 9)

Reads data/embeddings.pkl, initializes the SQLite database (database/attendance.db),
and migrates all person records and face embeddings into SQLite.
"""

import os
import pickle
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import add_person, init_db


def main() -> None:
    """Migrate known face embeddings from data/embeddings.pkl to SQLite database."""
    embeddings_file = os.path.join(PROJECT_ROOT, "data", "embeddings.pkl")

    # 1. Check if embeddings.pkl exists
    if not os.path.exists(embeddings_file):
        print(f"Error: Embeddings file '{embeddings_file}' not found.")
        print("Please run Stage 6 (python src/encode_faces.py) first.")
        sys.exit(1)

    # 2. Load embeddings dictionary from pickle file
    print(f"Loading embeddings from '{embeddings_file}'...")
    with open(embeddings_file, "rb") as f:
        known_embeddings = pickle.load(f)

    if not known_embeddings:
        print(f"Error: Embeddings file '{embeddings_file}' is empty.")
        sys.exit(1)

    # 3. Initialize SQLite database tables
    print("Initializing SQLite database structure...")
    init_db()

    # 4. Migrate each person + embedding into SQLite
    migrated_count = 0
    for person_name, embedding in known_embeddings.items():
        if add_person(person_name, embedding):
            migrated_count += 1
            print(f"Migrated: {person_name}")

    # 5. Print summary output
    db_rel_path = os.path.join("database", "attendance.db")
    print(f"\nMigrated {migrated_count} people to {db_rel_path}")


if __name__ == "__main__":
    main()
