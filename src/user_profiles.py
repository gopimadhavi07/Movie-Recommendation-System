"""
src/user_profiles.py
====================
SQLite Database Manager for User Profiles (Phase 7).
Handles user registration, user ratings, and watch history tracking.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from src.config import DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("movie_rec.user_profiles")

DB_PATH = DATA_DIR / "users.db"

def init_db():
    """Initializes the SQLite database schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create Users Table (Local Application Users start from ID 10000 to avoid MovieLens clash)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # In SQLite, AUTOINCREMENT starts from 1. We want custom users to start at 100000.
    # We can handle this by inserting a dummy row if table is empty and resetting sqlite_sequence, 
    # but simpler to just assign it in Python or let it auto-increment starting from max(movielens_id) + 1.
    
    # Create Ratings Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_ratings (
            rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, movie_id)
        )
    """)
    
    # Create Watch History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watch_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, movie_id)
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"User profile database initialized at {DB_PATH}")

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def create_user(username: str) -> int:
    """Creates a new user and returns their user_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if exists
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            return row[0]
            
        cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
        conn.commit()
        
        user_id = cursor.lastrowid
        # If user_id is small (overlaps with MovieLens), shift it. MovieLens small has 610 users.
        if user_id < 100000:
            new_id = 100000 + user_id
            cursor.execute("UPDATE users SET user_id = ? WHERE user_id = ?", (new_id, user_id))
            conn.commit()
            user_id = new_id
            
        logger.info(f"Created new user '{username}' with ID {user_id}")
        return user_id
    except sqlite3.IntegrityError:
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        return cursor.fetchone()[0]
    finally:
        conn.close()

def get_all_users() -> pd.DataFrame:
    """Returns a DataFrame of all registered users."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()
    return df

def add_rating(user_id: int, movie_id: int, rating: float):
    """Adds or updates a user rating for a movie, and also marks it as watched."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO user_ratings (user_id, movie_id, rating) 
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, movie_id) 
            DO UPDATE SET rating=excluded.rating, timestamp=CURRENT_TIMESTAMP
        """, (user_id, movie_id, rating))
        
        # Also add to watch history implicitly
        cursor.execute("""
            INSERT OR IGNORE INTO watch_history (user_id, movie_id) VALUES (?, ?)
        """, (user_id, movie_id))
        
        conn.commit()
        logger.info(f"User {user_id} rated movie {movie_id} -> {rating}")
    finally:
        conn.close()

def add_to_watch_history(user_id: int, movie_id: int):
    """Marks a movie as watched by the user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO watch_history (user_id, movie_id) VALUES (?, ?)
        """, (user_id, movie_id))
        conn.commit()
    finally:
        conn.close()

def get_user_ratings(user_id: int) -> pd.DataFrame:
    """Returns all ratings for a user."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT movie_id, rating, timestamp FROM user_ratings WHERE user_id = ?", conn, params=(user_id,))
    conn.close()
    return df

def get_watch_history(user_id: int) -> list:
    """Returns list of movie IDs watched by the user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT movie_id FROM watch_history WHERE user_id = ?", (user_id,))
    watched = [r[0] for r in cursor.fetchall()]
    conn.close()
    return watched

# Initialize DB on import
init_db()
