# db.py
from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterable, Literal, Optional, Sequence, Tuple, Union

import mysql.connector
from mysql.connector import pooling
from mysql.connector.connection import MySQLConnection

import os
from dotenv import load_dotenv


try:
    import streamlit as st
    HAS_ST = True
except Exception:
    HAS_ST = False

# ---------- Connection Pool ----------
load_dotenv()

def _get_mysql_config() -> dict:
    return {
        "host": os.getenv("TIDB_HOST", "127.0.0.1"),
        "port": int(os.getenv("TIDB_PORT", "4000")),
        "user": os.getenv("TIDB_USER", "root"),
        "password": os.getenv("TIDB_PASSWORD", ""),
        "database": os.getenv("TIDB_DB_NAME", "test"),
    }

# Example pooled connection
def get_pool() -> pooling.MySQLConnectionPool:
    cfg = _get_mysql_config()
    return pooling.MySQLConnectionPool(
        pool_name="app_pool",
        pool_size=5,
        pool_reset_session=True,
        autocommit=False,
        **cfg,
    )

def get_connection(autocommit: bool = False) -> MySQLConnection:
    pool = get_pool()
    conn: MySQLConnection = pool.get_connection()
    conn.autocommit = autocommit
    return conn


@contextmanager
def connect(autocommit: bool = False):
    """
    Context manager that yields a connection and guarantees close/rollback.
    Usage:
        with connect() as conn:
            ...
    """
    conn = get_connection(autocommit=autocommit)
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if conn.is_connected() and not autocommit:
            conn.rollback()
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

@contextmanager
def cursor(conn: MySQLConnection, dictionary: bool = True):
    """
    Context manager for a cursor. Defaults to dict rows.
    """
    cur = conn.cursor(dictionary=dictionary)
    try:
        yield cur
    finally:
        try:
            cur.close()
        except Exception:
            pass

# ---------- Generic Helpers ----------

FetchType = Literal["none", "one", "all"]

def execute_sql(
    sql: str,
    params: Optional[Union[Sequence[Any], Tuple[Any, ...]]] = None,
    *,
    fetch: FetchType = "none",
    many: bool = False,
    autocommit: bool = False,
    dictionary: bool = True,
) -> Union[None, dict, list[dict], int]:
    """
    Execute a parameterized SQL statement safely.

    Returns:
      - fetch="none": affected row count (int)
      - fetch="one":  single row (dict or None)
      - fetch="all":  list of rows (list[dict])

    Example:
      execute_sql("INSERT INTO t(a,b) VALUES (%s,%s)", (1,2))
      rows = execute_sql("SELECT * FROM t WHERE id=%s", (5,), fetch="all")
    """
    with connect(autocommit=autocommit) as conn:
        with cursor(conn, dictionary=dictionary) as cur:
            if many:
                if not isinstance(params, Iterable) or isinstance(params, (bytes, str)):
                    raise ValueError("For many=True, params must be an iterable of tuples/sequences.")
                cur.executemany(sql, params)  # type: ignore[arg-type]
            else:
                cur.execute(sql, params)

            if fetch == "none":
                return cur.rowcount
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            raise ValueError("Invalid fetch type")

def insert_and_get_id(sql: str, params: Sequence[Any]) -> int:
    """
    Run an INSERT and return lastrowid.
    """
    with connect(autocommit=False) as conn:
        with cursor(conn, dictionary=True) as cur:
            cur.execute(sql, params)
            last_id = cur.lastrowid
        # commit in connect()
    return int(last_id)

def to_dataframe(sql: str, params: Optional[Sequence[Any]] = None):
    """
    Load a SELECT into a pandas DataFrame.
    """
    import pandas as pd
    with connect(autocommit=True) as conn:
        # Pandas can use DBAPI connections directly
        return pd.read_sql(sql, con=conn, params=params)

# ---------- Schema & Example Domain (Job Applications) ----------

CREATE_JOB_APPS_TABLE = """
CREATE TABLE IF NOT EXISTS job_applications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    job_title VARCHAR(255) NOT NULL,
    company VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    application_date DATE,
    status ENUM('Planned','Saved','Applied','OA/Challenge','Interviewing','Offer','Rejected','Withdrawn') DEFAULT 'Applied',
    job_link TEXT,
    salary_min INT NULL,
    salary_max INT NULL,
    contact_name VARCHAR(255),
    contact_email VARCHAR(255),
    job_description MEDIUMTEXT,
    resume_version VARCHAR(255),
    cover_letter TINYINT(1) DEFAULT 0,
    next_steps TEXT,
    notes TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

def ensure_schema() -> None:
    execute_sql(CREATE_JOB_APPS_TABLE)

def add_application(app: dict) -> int:
    """
    Insert a job application. Returns new id.
    Required keys: job_title, company
    """
    sql = """
    INSERT INTO job_applications
    (job_title, company, location, application_date, status, job_link, salary_min, salary_max,
     contact_name, contact_email, job_description, resume_version, cover_letter, next_steps, notes)
    VALUES
    (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    params = (
        app.get("job_title"),
        app.get("company"),
        app.get("location"),
        app.get("application_date"),   # 'YYYY-MM-DD' or date object (mysql.connector will convert)
        app.get("status", "Applied"),
        app.get("job_link"),
        app.get("salary_min"),
        app.get("salary_max"),
        app.get("contact_name"),
        app.get("contact_email"),
        app.get("job_description"),
        app.get("resume_version"),
        1 if app.get("cover_letter") in (True, "Yes", "yes", 1) else 0,
        app.get("next_steps"),
        app.get("notes"),
    )
    return insert_and_get_id(sql, params)

def get_application(app_id: int) -> Optional[dict]:
    return execute_sql(
        "SELECT * FROM job_applications WHERE id=%s",
        (app_id,),
        fetch="one",
    )

def list_applications(limit: int = 200, offset: int = 0) -> list[dict]:
    return execute_sql(
        "SELECT * FROM job_applications ORDER BY created_at DESC LIMIT %s OFFSET %s",
        (limit, offset),
        fetch="all",
    )

def update_status(app_id: int, status: str) -> int:
    return execute_sql(
        "UPDATE job_applications SET status=%s WHERE id=%s",
        (status, app_id),
        fetch="none",
    )

def delete_application(app_id: int) -> int:
    return execute_sql(
        "DELETE FROM job_applications WHERE id=%s",
        (app_id,),
        fetch="none",
    )

# --- Experiences schema & helpers --------------------------------------------

CREATE_EXPERIENCES_TABLE = """
CREATE TABLE IF NOT EXISTS experiences (
    id INT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    exp_type ENUM('Internship','Project','Work','Leadership','Research','Volunteer','Other') NOT NULL,
    organization VARCHAR(255) NOT NULL,
    role_title VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    start_date DATE NOT NULL,
    end_date DATE NULL,
    is_current TINYINT(1) DEFAULT 0,
    hours_per_week INT NULL,
    link TEXT,
    technologies VARCHAR(255),
    description MEDIUMTEXT NOT NULL,
    impact MEDIUMTEXT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

def ensure_experiences_schema() -> None:
    execute_sql(CREATE_EXPERIENCES_TABLE)

def add_experience(exp: dict) -> int:
    sql = """
    INSERT INTO experiences
    (exp_type, organization, role_title, location, start_date, end_date, is_current,
     hours_per_week, link, technologies, description, impact)
    VALUES
    (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    params = (
        exp["exp_type"],
        exp["organization"],
        exp["role_title"],
        exp.get("location"),
        exp["start_date"],     # date object OK
        exp.get("end_date"),
        1 if exp.get("is_current") else 0,
        exp.get("hours_per_week"),
        exp.get("link"),
        exp.get("technologies"),
        exp["description"],
        exp.get("impact"),
    )
    return insert_and_get_id(sql, params)

def list_experiences(limit: int = 500, offset: int = 0) -> list[dict]:
    return execute_sql(
        "SELECT * FROM experiences ORDER BY start_date DESC, id DESC LIMIT %s OFFSET %s",
        (limit, offset),
        fetch="all",
    )

def get_experience(exp_id: int) -> dict | None:
    return execute_sql("SELECT * FROM experiences WHERE id=%s", (exp_id,), fetch="one")

def update_experience_impact(exp_id: int, impact: str) -> int:
    return execute_sql(
        "UPDATE experiences SET impact=%s WHERE id=%s",
        (impact, exp_id),
        fetch="none",
    )

def update_experience_basic(exp_id: int, fields: dict) -> int:
    # update selected basic fields (optional utility)
    allowed = {
        "exp_type","organization","role_title","location","start_date",
        "end_date","is_current","hours_per_week","link","technologies","description"
    }
    sets, params = [], []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k}=%s")
            params.append(v)
    if not sets:
        return 0
    params.append(exp_id)
    return execute_sql(f"UPDATE experiences SET {', '.join(sets)} WHERE id=%s", tuple(params), fetch="none")

def delete_experience(exp_id: int) -> int:
    return execute_sql("DELETE FROM experiences WHERE id=%s", (exp_id,), fetch="none")

# --- Background schema & helpers ---------------------------------------------

CREATE_BACKGROUND_TABLE = """
CREATE TABLE IF NOT EXISTS user_background (
    id INT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    education_level VARCHAR(255),
    coursework_projects MEDIUMTEXT,
    skills_experiences MEDIUMTEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

def ensure_background_schema() -> None:
    execute_sql(CREATE_BACKGROUND_TABLE)

def add_background(bg: dict) -> int:
    sql = """
    INSERT INTO user_background (education_level, coursework_projects, skills_experiences)
    VALUES (%s,%s,%s)
    """
    return insert_and_get_id(sql, (
        bg.get("education_level"),
        bg.get("coursework_projects"),
        bg.get("skills_experiences"),
    ))

def get_latest_background() -> dict | None:
    return execute_sql(
        "SELECT * FROM user_background ORDER BY updated_at DESC, id DESC LIMIT 1",
        fetch="one",
    )

def upsert_background(bg: dict) -> int:
    """Insert first background or update the latest one. Returns affected/inserted id."""
    existing = get_latest_background()
    if not existing:
        return add_background(bg)
    # update existing
    return execute_sql(
        "UPDATE user_background SET education_level=%s, coursework_projects=%s, skills_experiences=%s WHERE id=%s",
        (
            bg.get("education_level"),
            bg.get("coursework_projects"),
            bg.get("skills_experiences"),
            existing["id"],
        ),
        fetch="none",
    )