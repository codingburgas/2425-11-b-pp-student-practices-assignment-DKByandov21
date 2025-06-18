#!/usr/bin/env python3
"""
Script to add feedback table to database manually
"""

import sqlite3
import os

def add_feedback_table():
    """Add feedback table to database"""
    
    # Path to the database
    db_path = os.path.join('instance', 'app.db')
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the feedback table already exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
        if cursor.fetchone():
            print("Dropping existing feedback table...")
            cursor.execute("DROP TABLE feedback")
        
        # Create the feedback table with correct structure
        cursor.execute("""
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rating INTEGER NOT NULL,
                comment TEXT,
                is_public BOOLEAN DEFAULT 1,
                user_id INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user (id)
            )
        """)
        conn.commit()
        print("Successfully created feedback table")
        
        # Verify the table structure
        cursor.execute("PRAGMA table_info(feedback)")
        columns = cursor.fetchall()
        print("\nFeedback table structure:")
        for column in columns:
            print(f"  {column[1]} ({column[2]})")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    add_feedback_table() 