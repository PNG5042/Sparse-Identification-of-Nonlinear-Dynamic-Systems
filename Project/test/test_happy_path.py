# tests/test_happy_path.py
import sqlite3
from Project.test.app import create_and_insert_demo, DB

def test_insert_demo(tmp_path, monkeypatch):
    msg = create_and_insert_demo("alice")
    assert msg == "inserted: alice"
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT name FROM users WHERE name='alice'")
    row = cur.fetchone()
    assert row[0] == "alice"
    conn.close()
