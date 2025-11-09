# app.py
import sqlite3
from pathlib import Path

DB = Path("demo.db")

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

def create_and_insert_demo(name="demo-user"):
    init_db()
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO users (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()
    return f"inserted: {name}"
