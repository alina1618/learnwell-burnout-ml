import sqlite3

from app.config import DB_PATH


def get_connection():
    """Відкриває з'єднання з students.db."""
    return sqlite3.connect(DB_PATH)


def create_tables():
    """Створює дві нові таблиці (якщо їх ще немає): predictions та inference_inputs."""
    conn = get_connection()
    cur = conn.cursor()

    # У лабі 3 вже була таблиця predictions, але з іншими колонками.
    # Щоб не змішувати, перейменовуємо стару на predictions_lab3.
    cur.execute("PRAGMA table_info(predictions)")
    old_columns = [row[1] for row in cur.fetchall()]
    if old_columns and "model_version" not in old_columns:
        cur.execute("ALTER TABLE predictions RENAME TO predictions_lab3")

    # Усі вхідні параметри з HTTP-запиту на передбачення
    cur.execute("""
        CREATE TABLE IF NOT EXISTS inference_inputs (
            id                         INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id                 TEXT,
            input_source               TEXT,
            Student_ID                 INTEGER,
            Major_Category             TEXT,
            Year_of_Study              TEXT,
            Pre_Semester_GPA           REAL,
            Weekly_GenAI_Hours         REAL,
            Primary_Use_Case           TEXT,
            Prompt_Engineering_Skill   TEXT,
            Tool_Diversity             INTEGER,
            Paid_Subscription          INTEGER,
            Traditional_Study_Hours    REAL,
            Perceived_AI_Dependency    INTEGER,
            Institutional_Policy       TEXT,
            Anxiety_Level_During_Exams INTEGER,
            Post_Semester_GPA          REAL,
            Skill_Retention_Score      REAL,
            timestamp                  TEXT,
            created_at                 TEXT
        )
    """)

    # Результати передбачень: source = 'train' або 'inference'
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            Student_ID         INTEGER,
            model_version      TEXT,
            source             TEXT CHECK (source IN ('train', 'inference')),
            predicted_code     INTEGER,
            predicted_label    TEXT,
            proba_low          REAL,
            proba_medium       REAL,
            proba_high         REAL,
            true_label         TEXT,
            timestamp          TEXT,
            inference_input_id INTEGER REFERENCES inference_inputs(id),
            created_at         TEXT
        )
    """)

    conn.commit()
    conn.close()