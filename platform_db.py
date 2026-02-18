"""Persistent platform database for accounts, projects, models, phrases, and events."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

DEFAULT_DB_PATH = "outputs/neuro_platform.db"


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _connect(db_path: str) -> sqlite3.Connection:
    _ensure_parent(db_path)
    conn = sqlite3.connect(db_path, factory=_ClosingConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> str:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                preferred_lang TEXT NOT NULL DEFAULT 'en',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                stack_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(owner_id, name),
                FOREIGN KEY (owner_id) REFERENCES accounts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                agent_name TEXT NOT NULL DEFAULT '',
                dsl TEXT NOT NULL DEFAULT '',
                checkpoint_path TEXT NOT NULL DEFAULT '',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS phrases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                language TEXT NOT NULL,
                phrase TEXT NOT NULL,
                frequency INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'seed',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(language, phrase)
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT NOT NULL DEFAULT 'global',
                title TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                url TEXT NOT NULL DEFAULT '',
                published_at TEXT NOT NULL DEFAULT '',
                ingested_at TEXT NOT NULL,
                UNIQUE(source, title, published_at)
            );

            CREATE TABLE IF NOT EXISTS api_sessions (
                id TEXT PRIMARY KEY,
                account_id INTEGER NOT NULL,
                token_hash TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0,
                meta_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                account_id INTEGER,
                run_mode TEXT NOT NULL,
                request_json TEXT NOT NULL DEFAULT '{}',
                response_json TEXT NOT NULL DEFAULT '{}',
                latency_ms REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES api_sessions(id) ON DELETE SET NULL,
                FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_id);
            CREATE INDEX IF NOT EXISTS idx_models_project ON models(project_id);
            CREATE INDEX IF NOT EXISTS idx_phrases_lang ON phrases(language);
            CREATE INDEX IF NOT EXISTS idx_events_region ON events(region);
            CREATE INDEX IF NOT EXISTS idx_api_sessions_account ON api_sessions(account_id);
            CREATE INDEX IF NOT EXISTS idx_api_sessions_active ON api_sessions(revoked, expires_at);
            CREATE INDEX IF NOT EXISTS idx_model_runs_session ON model_runs(session_id);
            CREATE INDEX IF NOT EXISTS idx_model_runs_account ON model_runs(account_id);
            """
        )
    return db_path


def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${digest}"


def _verify_password(stored_hash: str, password: str) -> bool:
    try:
        salt, expected = stored_hash.split("$", 1)
    except ValueError:
        return False
    actual = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return secrets.compare_digest(expected, actual)


def create_account(
    username: str,
    password: str,
    db_path: str = DEFAULT_DB_PATH,
    role: str = "user",
    preferred_lang: str = "en",
) -> dict:
    if not username or not password:
        raise ValueError("username and password are required")
    init_db(db_path)
    created_at = _utc_now()
    with _connect(db_path) as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO accounts (username, password_hash, role, preferred_lang, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username.strip(), _hash_password(password), role, preferred_lang, created_at),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"account '{username}' already exists") from exc
        account_id = int(cur.lastrowid)
    return {
        "id": account_id,
        "username": username.strip(),
        "role": role,
        "preferred_lang": preferred_lang,
        "created_at": created_at,
    }


def authenticate_account(username: str, password: str, db_path: str = DEFAULT_DB_PATH) -> dict:
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, preferred_lang, created_at FROM accounts WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if not row or not _verify_password(str(row["password_hash"]), password):
        return {"ok": False}
    return {
        "ok": True,
        "id": int(row["id"]),
        "username": str(row["username"]),
        "role": str(row["role"]),
        "preferred_lang": str(row["preferred_lang"]),
        "created_at": str(row["created_at"]),
    }


def list_accounts(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, username, role, preferred_lang, created_at FROM accounts ORDER BY id ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def _resolve_owner_id(conn: sqlite3.Connection, owner: str) -> int:
    row = conn.execute("SELECT id FROM accounts WHERE username = ?", (owner.strip(),)).fetchone()
    if not row:
        raise ValueError(f"unknown account '{owner}'")
    return int(row["id"])


def create_project(
    owner_username: str,
    name: str,
    description: str = "",
    stack_json: str = "[]",
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    if not owner_username or not name:
        raise ValueError("owner_username and name are required")
    init_db(db_path)
    created_at = _utc_now()
    with _connect(db_path) as conn:
        owner_id = _resolve_owner_id(conn, owner_username)
        try:
            cur = conn.execute(
                """
                INSERT INTO projects (owner_id, name, description, stack_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (owner_id, name.strip(), description, stack_json, created_at, created_at),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"project '{name}' already exists for owner '{owner_username}'") from exc
        project_id = int(cur.lastrowid)
    return {
        "id": project_id,
        "owner": owner_username.strip(),
        "name": name.strip(),
        "description": description,
        "stack_json": stack_json,
        "created_at": created_at,
    }


def list_projects(db_path: str = DEFAULT_DB_PATH, owner_username: str = "") -> list[dict]:
    init_db(db_path)
    sql = """
        SELECT p.id, a.username AS owner, p.name, p.description, p.stack_json, p.created_at, p.updated_at
        FROM projects p
        JOIN accounts a ON p.owner_id = a.id
    """
    params: tuple = ()
    if owner_username:
        sql += " WHERE a.username = ?"
        params = (owner_username.strip(),)
    sql += " ORDER BY p.id ASC"

    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _resolve_project_id(conn: sqlite3.Connection, project_name: str, owner_username: str = "") -> int:
    if owner_username:
        row = conn.execute(
            """
            SELECT p.id
            FROM projects p
            JOIN accounts a ON p.owner_id = a.id
            WHERE p.name = ? AND a.username = ?
            """,
            (project_name.strip(), owner_username.strip()),
        ).fetchone()
    else:
        row = conn.execute("SELECT id FROM projects WHERE name = ? ORDER BY id ASC LIMIT 1", (project_name.strip(),)).fetchone()
    if not row:
        raise ValueError(f"unknown project '{project_name}'")
    return int(row["id"])


def register_model(
    project_name: str,
    model_name: str,
    checkpoint_path: str = "",
    dsl: str = "",
    metrics_json: str = "{}",
    agent_name: str = "",
    db_path: str = DEFAULT_DB_PATH,
    owner_username: str = "",
) -> dict:
    if not project_name or not model_name:
        raise ValueError("project_name and model_name are required")
    init_db(db_path)
    created_at = _utc_now()
    with _connect(db_path) as conn:
        project_id = _resolve_project_id(conn, project_name, owner_username=owner_username)
        cur = conn.execute(
            """
            INSERT INTO models (project_id, name, agent_name, dsl, checkpoint_path, metrics_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (project_id, model_name.strip(), agent_name, dsl, checkpoint_path, metrics_json, created_at),
        )
        model_id = int(cur.lastrowid)
    return {
        "id": model_id,
        "project": project_name.strip(),
        "name": model_name.strip(),
        "agent_name": agent_name,
        "checkpoint_path": checkpoint_path,
        "created_at": created_at,
    }


def list_models(db_path: str = DEFAULT_DB_PATH, project_name: str = "", owner_username: str = "") -> list[dict]:
    init_db(db_path)
    sql = """
        SELECT m.id, p.name AS project, a.username AS owner, m.name, m.agent_name, m.dsl, m.checkpoint_path, m.metrics_json, m.created_at
        FROM models m
        JOIN projects p ON m.project_id = p.id
        JOIN accounts a ON p.owner_id = a.id
    """
    params: tuple = ()
    if project_name and owner_username:
        sql += " WHERE p.name = ? AND a.username = ?"
        params = (project_name.strip(), owner_username.strip())
    elif project_name:
        sql += " WHERE p.name = ?"
        params = (project_name.strip(),)
    sql += " ORDER BY m.id ASC"
    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def ensure_project(
    owner_username: str,
    name: str,
    description: str = "",
    stack_json: str = "[]",
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    projects = list_projects(db_path=db_path, owner_username=owner_username)
    for proj in projects:
        if str(proj.get("name", "")).strip() == name.strip():
            return proj
    return create_project(
        owner_username=owner_username,
        name=name,
        description=description,
        stack_json=stack_json,
        db_path=db_path,
    )


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_api_session(
    username: str,
    password: str,
    db_path: str = DEFAULT_DB_PATH,
    ttl_hours: int = 24,
    meta_json: str = "{}",
) -> dict:
    auth = authenticate_account(username=username, password=password, db_path=db_path)
    if not auth.get("ok"):
        return {"ok": False}
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=max(1, int(ttl_hours)))
    session_id = secrets.token_hex(12)
    token = secrets.token_urlsafe(32)
    token_hash = _hash_token(token)
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO api_sessions
            (id, account_id, token_hash, created_at, last_seen_at, expires_at, revoked, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                session_id,
                int(auth["id"]),
                token_hash,
                now.isoformat(),
                now.isoformat(),
                expires.isoformat(),
                meta_json,
            ),
        )
    return {
        "ok": True,
        "session_id": session_id,
        "token": token,
        "username": str(auth["username"]),
        "role": str(auth["role"]),
        "expires_at": expires.isoformat(),
    }


def authenticate_api_token(token: str, db_path: str = DEFAULT_DB_PATH, touch: bool = True) -> dict:
    if not token:
        return {"ok": False, "error": "missing token"}
    init_db(db_path)
    tok_hash = _hash_token(token)
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT s.id AS session_id, s.account_id, s.created_at, s.last_seen_at, s.expires_at, s.revoked,
                   a.username, a.role, a.preferred_lang
            FROM api_sessions s
            JOIN accounts a ON s.account_id = a.id
            WHERE s.token_hash = ?
            """,
            (tok_hash,),
        ).fetchone()
        if not row:
            return {"ok": False, "error": "invalid token"}
        if int(row["revoked"]) != 0:
            return {"ok": False, "error": "session revoked"}
        exp = str(row["expires_at"])
        if exp and exp < _utc_now():
            return {"ok": False, "error": "session expired"}
        if touch:
            conn.execute(
                "UPDATE api_sessions SET last_seen_at = ? WHERE id = ?",
                (_utc_now(), str(row["session_id"])),
            )
    return {
        "ok": True,
        "session_id": str(row["session_id"]),
        "account_id": int(row["account_id"]),
        "username": str(row["username"]),
        "role": str(row["role"]),
        "preferred_lang": str(row["preferred_lang"]),
        "expires_at": str(row["expires_at"]),
    }


def revoke_api_token(token: str, db_path: str = DEFAULT_DB_PATH) -> dict:
    if not token:
        return {"ok": False, "error": "missing token"}
    init_db(db_path)
    tok_hash = _hash_token(token)
    with _connect(db_path) as conn:
        conn.execute("UPDATE api_sessions SET revoked = 1 WHERE token_hash = ?", (tok_hash,))
        row = conn.execute("SELECT changes() AS c").fetchone()
    if not row or int(row["c"]) <= 0:
        return {"ok": False, "error": "token not found"}
    return {"ok": True}


def list_api_sessions(db_path: str = DEFAULT_DB_PATH, username: str = "", limit: int = 100) -> list[dict]:
    init_db(db_path)
    sql = """
        SELECT s.id, a.username, a.role, s.created_at, s.last_seen_at, s.expires_at, s.revoked, s.meta_json
        FROM api_sessions s
        JOIN accounts a ON s.account_id = a.id
    """
    params: tuple = ()
    if username:
        sql += " WHERE a.username = ?"
        params = (username.strip(),)
    sql += " ORDER BY s.created_at DESC LIMIT ?"
    params = params + (int(max(1, limit)),)
    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def log_model_run(
    run_mode: str,
    request_obj: dict,
    response_obj: dict,
    latency_ms: float,
    db_path: str = DEFAULT_DB_PATH,
    session_id: str = "",
    account_id: int | None = None,
) -> dict:
    init_db(db_path)
    created_at = _utc_now()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO model_runs
            (session_id, account_id, run_mode, request_json, response_json, latency_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id or None,
                account_id if account_id is not None else None,
                run_mode.strip() or "unknown",
                json.dumps(request_obj or {}, ensure_ascii=True),
                json.dumps(response_obj or {}, ensure_ascii=True),
                float(max(0.0, latency_ms)),
                created_at,
            ),
        )
        run_id = int(cur.lastrowid)
    return {"id": run_id, "created_at": created_at}


def list_model_runs(
    db_path: str = DEFAULT_DB_PATH,
    username: str = "",
    limit: int = 200,
) -> list[dict]:
    init_db(db_path)
    sql = """
        SELECT r.id, r.session_id, r.account_id, a.username, r.run_mode, r.request_json, r.response_json, r.latency_ms, r.created_at
        FROM model_runs r
        LEFT JOIN accounts a ON r.account_id = a.id
    """
    params: tuple = ()
    if username:
        sql += " WHERE a.username = ?"
        params = (username.strip(),)
    sql += " ORDER BY r.id DESC LIMIT ?"
    params = params + (int(max(1, limit)),)
    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def upsert_phrase(
    language: str,
    phrase: str,
    frequency: int = 1,
    source: str = "seed",
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    if not language or not phrase:
        return
    init_db(db_path)
    now = _utc_now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO phrases (language, phrase, frequency, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(language, phrase) DO UPDATE SET
                frequency = phrases.frequency + excluded.frequency,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (language.strip(), phrase.strip(), int(max(1, frequency)), source, now, now),
        )


def bulk_upsert_phrases(entries: Iterable[dict], db_path: str = DEFAULT_DB_PATH) -> int:
    count = 0
    for entry in entries:
        upsert_phrase(
            language=str(entry.get("language", "en")),
            phrase=str(entry.get("phrase", "")).strip(),
            frequency=int(entry.get("frequency", 1)),
            source=str(entry.get("source", "seed")),
            db_path=db_path,
        )
        count += 1
    return count


def list_phrases(db_path: str = DEFAULT_DB_PATH, language: str = "", limit: int = 100) -> list[dict]:
    init_db(db_path)
    sql = "SELECT language, phrase, frequency, source, created_at, updated_at FROM phrases"
    params: tuple = ()
    if language:
        sql += " WHERE language = ?"
        params = (language.strip(),)
    sql += " ORDER BY frequency DESC, phrase ASC LIMIT ?"
    params = params + (int(max(1, limit)),)
    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def add_events(events: Iterable[dict], db_path: str = DEFAULT_DB_PATH) -> int:
    init_db(db_path)
    inserted = 0
    now = _utc_now()
    with _connect(db_path) as conn:
        for event in events:
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO events
                    (region, title, summary, source, url, published_at, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(event.get("region", "global")),
                        str(event.get("title", "")).strip(),
                        str(event.get("summary", "")),
                        str(event.get("source", "")),
                        str(event.get("url", "")),
                        str(event.get("published_at", "")),
                        now,
                    ),
                )
                # sqlite3 total_changes is connection-wide; compare count via changes().
                row = conn.execute("SELECT changes() AS c").fetchone()
                inserted += int(row["c"]) if row else 0
            except Exception:
                continue
    return inserted


def list_events(db_path: str = DEFAULT_DB_PATH, limit: int = 100) -> list[dict]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT region, title, summary, source, url, published_at, ingested_at
            FROM events
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(max(1, limit)),),
        ).fetchall()
    return [dict(r) for r in rows]


def seed_common_phrases(db_path: str = DEFAULT_DB_PATH) -> int:
    seeds = [
        {"language": "en", "phrase": "machine learning", "frequency": 10},
        {"language": "en", "phrase": "breaking news", "frequency": 10},
        {"language": "en", "phrase": "global markets", "frequency": 8},
        {"language": "es", "phrase": "aprendizaje automatico", "frequency": 10},
        {"language": "es", "phrase": "noticias de ultima hora", "frequency": 8},
        {"language": "fr", "phrase": "intelligence artificielle", "frequency": 10},
        {"language": "fr", "phrase": "actualites mondiales", "frequency": 8},
        {"language": "de", "phrase": "kunstliche intelligenz", "frequency": 10},
        {"language": "de", "phrase": "weltnachrichten", "frequency": 8},
        {"language": "pt", "phrase": "inteligencia artificial", "frequency": 10},
        {"language": "pt", "phrase": "eventos globais", "frequency": 8},
        {"language": "it", "phrase": "intelligenza artificiale", "frequency": 8},
        {"language": "hi", "phrase": "krtrim buddhimatta", "frequency": 8},
        {"language": "ar", "phrase": "thka astnaay", "frequency": 8},
        {"language": "ja", "phrase": "kikai gakushu", "frequency": 8},
        {"language": "ko", "phrase": "ingong jineung", "frequency": 8},
        {"language": "zh", "phrase": "ren gong zhi neng", "frequency": 8},
    ]
    return bulk_upsert_phrases(seeds, db_path=db_path)


def get_snapshot(db_path: str = DEFAULT_DB_PATH) -> dict:
    init_db(db_path)
    with _connect(db_path) as conn:
        accounts = int(conn.execute("SELECT COUNT(*) AS c FROM accounts").fetchone()["c"])
        projects = int(conn.execute("SELECT COUNT(*) AS c FROM projects").fetchone()["c"])
        models = int(conn.execute("SELECT COUNT(*) AS c FROM models").fetchone()["c"])
        phrases = int(conn.execute("SELECT COUNT(*) AS c FROM phrases").fetchone()["c"])
        events = int(conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"])
        sessions = int(conn.execute("SELECT COUNT(*) AS c FROM api_sessions").fetchone()["c"])
        runs = int(conn.execute("SELECT COUNT(*) AS c FROM model_runs").fetchone()["c"])
    return {
        "accounts": accounts,
        "projects": projects,
        "models": models,
        "phrases": phrases,
        "events": events,
        "sessions": sessions,
        "runs": runs,
    }
