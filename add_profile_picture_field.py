#!/usr/bin/env python3
"""
Script to add profile_picture field to User table manually
"""

import sqlite3
import os

def add_profile_picture_field():
    """Add profile_picture column to User table"""
    
    # Path to the database
    db_path = os.path.join('instance', 'app.db')
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the column already exists
        cursor.execute("PRAGMA table_info(user)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'profile_picture' not in columns:
            # Add the profile_picture column
            cursor.execute("ALTER TABLE user ADD COLUMN profile_picture VARCHAR(255)")
            conn.commit()
            print("Successfully added profile_picture column to User table")
        else:
            print("profile_picture column already exists")
        
        # Verify the column was added
        cursor.execute("PRAGMA table_info(user)")
        columns = cursor.fetchall()
        print("\nCurrent User table structure:")
        for column in columns:
            print(f"  {column[1]} ({column[2]})")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    add_profile_picture_field() 