"""SQLite storage and migration of the previously supplied course database."""
import json, sqlite3
from datetime import datetime
from config import DB_PATH
from syllabus_utils import split_lines

def db_conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS courses(
        id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, title TEXT, description TEXT,
        outcomes TEXT, topics TEXT, source TEXT, created_at TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS analyses(
        id INTEGER PRIMARY KEY AUTOINCREMENT, proposed_label TEXT, compared_count INTEGER,
        best_match TEXT, best_similarity REAL, novelty REAL, created_at TEXT)""")
    con.commit()
    migrate_legacy_analysis_table(con)
    return con

def migrate_legacy_analysis_table(con):
    cols = [r[1] for r in con.execute("PRAGMA table_info(analyses)").fetchall()]
    required = {"id", "proposed_label", "compared_count", "best_match", "best_similarity", "novelty", "created_at"}
    if cols and not required.issubset(set(cols)):
        con.execute("ALTER TABLE analyses RENAME TO legacy_analyses")
        con.execute("""CREATE TABLE analyses(
            id INTEGER PRIMARY KEY AUTOINCREMENT, proposed_label TEXT, compared_count INTEGER,
            best_match TEXT, best_similarity REAL, novelty REAL, created_at TEXT)""")
        con.commit()

def migrate_legacy_courses(con):
    """Import courses from legacy schema if the database still has its old JSON data column."""
    cols = [r[1] for r in con.execute("PRAGMA table_info(courses)").fetchall()]
    if "data" not in cols:
        return
    rows = con.execute("SELECT code,title,approved,data FROM courses").fetchall()
    con.execute("ALTER TABLE courses RENAME TO legacy_courses")
    con.execute("""CREATE TABLE courses(
        id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, title TEXT, description TEXT,
        outcomes TEXT, topics TEXT, source TEXT, created_at TEXT)""")
    for code, title, approved, raw in rows:
        try:
            item = json.loads(raw or "{}")
        except Exception:
            item = {}
        con.execute("""INSERT INTO courses(code,title,description,outcomes,topics,source,created_at)
            VALUES(?,?,?,?,?,?,?)""", (
            item.get("code", code or ""), item.get("title", title or ""),
            item.get("description", ""), "\n".join(item.get("outcomes", [])),
            "\n".join(item.get("topics", [])), f"Imported legacy catalog ({approved or 'existing'})",
            datetime.now().isoformat(timespec="seconds")))
    con.commit()

def load_courses(con):
    rows = con.execute("SELECT id,code,title,description,outcomes,topics,source FROM courses ORDER BY id").fetchall()
    return [{"id": r[0], "code": r[1] or "", "title": r[2] or "", "description": r[3] or "",
             "outcomes": split_lines(r[4]), "topics": split_lines(r[5]), "source": r[6] or ""} for r in rows]

def add_course(con, item, source="manual"):
    con.execute("""INSERT INTO courses(code,title,description,outcomes,topics,source,created_at)
        VALUES(?,?,?,?,?,?,?)""", (
        item.get("code", ""), item.get("title", ""), item.get("description", ""),
        "\n".join(item.get("outcomes", [])), "\n".join(item.get("topics", [])),
        source, datetime.now().isoformat(timespec="seconds")))
    con.commit()

def save_analysis(con, proposed, results):
    best = results[0] if results else {}
    con.execute("""INSERT INTO analyses(proposed_label,compared_count,best_match,best_similarity,novelty,created_at)
        VALUES(?,?,?,?,?,?)""", (proposed.get("title", ""), len(results), best.get("course", ""),
        best.get("similarity", 0), best.get("novelty", 1), datetime.now().isoformat(timespec="seconds")))
    con.commit()
