import os

import sqlite3
from datetime import datetime


# Load environment variables

def view_feedback():
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    results=cursor.execute("select * from  feedback")
    for feedback in results.fetchall():
        print(feedback)
        print("\n")
    conn.close()

# Run app
if __name__ == '__main__':
    view_feedback()