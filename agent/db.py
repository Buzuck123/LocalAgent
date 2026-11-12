import json
import hashlib
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .config import settings

_lock = threading.RLock()
_debug_setting_cache: dict[str, bool] = {}
_debug_setting_cache_at = 0.0


class _ClosingConnection(sqlite3.Connection):
    """Make `with connect()` close the SQLite handle after commit/rollback."""
    def __exit__(self, exc_type, exc, traceback):
        try:
            return super().__exit__(exc_type, exc, traceback)
        finally:
            self.close()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def connect():
    con = sqlite3.connect(str(settings.database_path), timeout=30, factory=_ClosingConnection)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    return con

def init_db():
    with _lock, connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL, attachments_json TEXT,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, user_message TEXT NOT NULL, attachments_json TEXT,
            status TEXT NOT NULL, result TEXT, error TEXT, checkpoint_json TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

        CREATE TABLE IF NOT EXISTS tool_operations (
            operation_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            arguments_json TEXT,
            status TEXT NOT NULL CHECK(status IN ('running', 'completed')),
            result_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tool_operations_task
            ON tool_operations(task_id);

        CREATE TABLE IF NOT EXISTS conversation_sandboxes (
            conversation_id TEXT PRIMARY KEY,
            container_id TEXT NOT NULL UNIQUE,
            container_name TEXT,
            host_port INTEGER NOT NULL,
            auth_token TEXT NOT NULL,
            session_name TEXT NOT NULL DEFAULT 'main',
            status TEXT NOT NULL DEFAULT 'running',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
            model TEXT NOT NULL,
            prompt_eval_count INTEGER,
            eval_count INTEGER,
            prompt_eval_duration_ns INTEGER,
            eval_duration_ns INTEGER,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_model_requests_task ON model_requests(task_id, id);
        CREATE TABLE IF NOT EXISTS debug_trace (
            event_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_debug_trace_task ON debug_trace(task_id, created_at, event_id);
        """)
        # Add attachment metadata without breaking existing databases.
        message_columns = {row[1] for row in con.execute("PRAGMA table_info(messages)").fetchall()}
        if "attachments_json" not in message_columns:
            con.execute("ALTER TABLE messages ADD COLUMN attachments_json TEXT")
        task_columns = {row[1] for row in con.execute("PRAGMA table_info(tasks)").fetchall()}
        if "attachments_json" not in task_columns:
            con.execute("ALTER TABLE tasks ADD COLUMN attachments_json TEXT")
        # Migrate databases created by older releases without requiring data loss.
        tool_columns = {row[1] for row in con.execute("PRAGMA table_info(tool_operations)").fetchall()}
        if "arguments_json" not in tool_columns:
            con.execute("ALTER TABLE tool_operations ADD COLUMN arguments_json TEXT")
        # SQLite cannot add a foreign key with ALTER COLUMN. Rebuild the two
        # journal tables transactionally, preserving all rows. Legacy orphaned
        # task references are retained as NULL (the task no longer exists).
        def has_task_cascade(table: str) -> bool:
            return any(
                row[2] == "tasks" and row[3] == "task_id"
                and row[4] == "id" and str(row[6]).upper() == "CASCADE"
                for row in con.execute(f"PRAGMA foreign_key_list({table})").fetchall()
            )

        if not has_task_cascade("tool_operations"):
            con.execute("""CREATE TABLE tool_operations_new (
                operation_id TEXT PRIMARY KEY, task_id TEXT, tool_name TEXT NOT NULL,
                arguments_json TEXT,
                status TEXT NOT NULL CHECK(status IN ('running', 'completed')),
                result_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
            )""")
            con.execute("""INSERT INTO tool_operations_new
                (operation_id, task_id, tool_name, arguments_json, status, result_json, created_at, updated_at)
                SELECT operation_id, CASE WHEN EXISTS(SELECT 1 FROM tasks t WHERE t.id=tool_operations.task_id)
                    THEN task_id ELSE NULL END, tool_name, arguments_json, status, result_json, created_at, updated_at
                FROM tool_operations""")
            con.execute("DROP TABLE tool_operations")
            con.execute("ALTER TABLE tool_operations_new RENAME TO tool_operations")
            con.execute("CREATE INDEX IF NOT EXISTS idx_tool_operations_task ON tool_operations(task_id)")

        if not has_task_cascade("model_requests"):
            con.execute("""CREATE TABLE model_requests_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, model TEXT NOT NULL,
                prompt_eval_count INTEGER, eval_count INTEGER,
                prompt_eval_duration_ns INTEGER, eval_duration_ns INTEGER, created_at TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
            )""")
            con.execute("""INSERT INTO model_requests_new
                (id, task_id, model, prompt_eval_count, eval_count, prompt_eval_duration_ns, eval_duration_ns, created_at)
                SELECT id, CASE WHEN task_id IS NOT NULL AND EXISTS(SELECT 1 FROM tasks t WHERE t.id=model_requests.task_id)
                    THEN task_id ELSE NULL END, model, prompt_eval_count, eval_count, prompt_eval_duration_ns, eval_duration_ns, created_at
                FROM model_requests""")
            con.execute("DROP TABLE model_requests")
            con.execute("ALTER TABLE model_requests_new RENAME TO model_requests")
            con.execute("CREATE INDEX IF NOT EXISTS idx_model_requests_task ON model_requests(task_id, id)")

        fk_errors = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk_errors:
            raise sqlite3.IntegrityError(f"Foreign-key migration produced violations: {len(fk_errors)}")

        # A process restart cannot safely resume an in-flight HTTP/tool call.
        # Mark interrupted tasks explicitly so users can inspect/retry them.
        con.execute("UPDATE tasks SET status='interrupted', error='Application restarted while task was running', updated_at=? WHERE status='running'", (now(),))

def get_sandbox(conversation_id: str) -> dict | None:
    with connect() as con:
        row = con.execute(
            "SELECT * FROM conversation_sandboxes WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()
        return dict(row) if row else None


def upsert_sandbox(conversation_id: str, container_id: str, container_name: str | None,
                   host_port: int, auth_token: str, session_name: str = "main",
                   status: str = "running") -> None:
    ts = now()
    with _lock, connect() as con:
        con.execute("""INSERT INTO conversation_sandboxes
            (conversation_id, container_id, container_name, host_port, auth_token,
             session_name, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
              container_id=excluded.container_id, container_name=excluded.container_name,
              host_port=excluded.host_port, auth_token=excluded.auth_token,
              session_name=excluded.session_name, status=excluded.status,
              updated_at=excluded.updated_at""",
            (conversation_id, container_id, container_name, host_port, auth_token,
             session_name, status, ts, ts))


def list_sandboxes() -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM conversation_sandboxes ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]


def update_sandbox_status(conversation_id: str, status: str) -> None:
    with _lock, connect() as con:
        con.execute("UPDATE conversation_sandboxes SET status=?, updated_at=? WHERE conversation_id=?",
                    (status, now(), conversation_id))


def delete_sandbox(conversation_id: str) -> bool:
    with _lock, connect() as con:
        cur = con.execute("DELETE FROM conversation_sandboxes WHERE conversation_id=?", (conversation_id,))
        return cur.rowcount > 0


def create_conversation(title: str = "Новый диалог") -> dict:
    cid, ts = str(uuid.uuid4()), now()
    with _lock, connect() as con:
        con.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)", (cid, title[:100], ts, ts))
    return get_conversation(cid)

def list_conversations() -> list[dict]:
    with connect() as con:
        rows = con.execute("""SELECT c.*,
          (SELECT content FROM messages m WHERE m.conversation_id=c.id ORDER BY id DESC LIMIT 1) AS last_message
          FROM conversations c ORDER BY updated_at DESC""").fetchall()
        return [dict(r) for r in rows]

def get_conversation(cid: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
        return dict(row) if row else None

def rename_conversation(cid: str, title: str):
    with _lock, connect() as con:
        con.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (title[:100], now(), cid),
        )

def delete_conversation(cid: str) -> bool:
    with _lock, connect() as con:
        cur = con.execute("DELETE FROM conversations WHERE id=?", (cid,))
        return cur.rowcount > 0

def add_message(cid: str, role: str, content: str, attachments: list[dict] | None = None) -> dict:
    ts = now()
    with _lock, connect() as con:
        cur = con.execute("INSERT INTO messages(conversation_id,role,content,created_at,attachments_json) VALUES (?,?,?,?,?)",
                          (cid, role, content, ts, json.dumps(attachments or [], ensure_ascii=False)))
        con.execute("UPDATE conversations SET updated_at=? WHERE id=?", (ts, cid))
        return dict(con.execute("SELECT * FROM messages WHERE id=?", (cur.lastrowid,)).fetchone())

def get_messages(cid: str, limit: int = 100) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM (SELECT * FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?) ORDER BY id",
                           (cid, limit)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try: item["attachments"] = json.loads(item.pop("attachments_json", "[]") or "[]")
            except (TypeError, json.JSONDecodeError): item["attachments"] = []
            result.append(item)
        return result

def get_conversation_attachments(cid: str) -> list[dict]:
    """Return all attachment metadata for a conversation, including old messages."""
    result: list[dict] = []
    with connect() as con:
        rows = con.execute(
            "SELECT attachments_json FROM messages WHERE conversation_id=? UNION ALL "
            "SELECT attachments_json FROM tasks WHERE conversation_id=?",
            (cid, cid),
        ).fetchall()
    for row in rows:
        try:
            items = json.loads(row["attachments_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(items, list):
            result.extend(item for item in items if isinstance(item, dict))
    return result


def create_task(cid: str, user_message: str, attachments: list[dict] | None = None) -> dict:
    tid, ts = str(uuid.uuid4()), now()
    with _lock, connect() as con:
        con.execute("""INSERT INTO tasks(id,conversation_id,user_message,attachments_json,status,created_at,updated_at)
                       VALUES(?,?,?,?,'queued',?,?)""", (tid, cid, user_message, json.dumps(attachments or [], ensure_ascii=False), ts, ts))
    return get_task(tid)

def update_task(tid: str, **fields):
    allowed = {"status", "result", "error", "checkpoint_json"}
    values = {k: v for k, v in fields.items() if k in allowed}
    values["updated_at"] = now()
    with _lock, connect() as con:
        con.execute("UPDATE tasks SET " + ", ".join(f"{k}=?" for k in values) + " WHERE id=?",
                    (*values.values(), tid))
    if "status" in values:
        record_debug_trace(tid, "task_status", {"status": values["status"], "error": fields.get("error")})

def get_task(tid: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
        if not row: return None
        item = dict(row)
        try: item["attachments"] = json.loads(item.pop("attachments_json", "[]") or "[]")
        except (TypeError, json.JSONDecodeError): item["attachments"] = []
        return item


def accept_task_clarification(tid: str, cid: str, checkpoint_json: str, answer: str, attachments: list[dict] | None = None) -> dict:
    """Atomically persist a clarification answer, checkpoint, and visible chat message."""
    if not answer or not answer.strip():
        raise ValueError("Ответ на уточнение не может быть пустым.")
    ts = now()
    with _lock, connect() as con:
        row = con.execute(
            "SELECT status, checkpoint_json FROM tasks WHERE id=? AND conversation_id=?",
            (tid, cid),
        ).fetchone()
        if row is None:
            raise ValueError("Задача для уточнения не найдена.")
        if row["status"] != "paused":
            raise ValueError("Задача больше не ожидает уточнения.")
        try:
            old_checkpoint = json.loads(row["checkpoint_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            old_checkpoint = {}
        old_clarification = old_checkpoint.get("clarification") if isinstance(old_checkpoint, dict) else None
        if not isinstance(old_checkpoint, dict) or old_checkpoint.get("pause_reason") != "needs_user_input" or (
            isinstance(old_clarification, dict) and old_clarification.get("resolved")
        ):
            raise ValueError("Задача больше не ожидает ответа на уточнение.")
        con.execute(
            "UPDATE tasks SET status='paused', checkpoint_json=?, error=NULL, updated_at=? WHERE id=?",
            (checkpoint_json, ts, tid),
        )
        con.execute(
            "INSERT INTO messages(conversation_id,role,content,created_at,attachments_json) VALUES (?,?,?,?,?)",
            (cid, "user", answer.strip(), ts, json.dumps(attachments or [], ensure_ascii=False)),
        )
        con.execute("UPDATE conversations SET updated_at=? WHERE id=?", (ts, cid))
    task = get_task(tid)
    if task is None:
        raise RuntimeError("Не удалось прочитать задачу после сохранения уточнения.")
    return task



def _tool_arguments_summary(arguments: dict[str, Any] | None) -> str:
    """Persist useful call parameters without copying large document payloads into the journal."""
    if not isinstance(arguments, dict):
        arguments = {}
    bulky_keys = {"content", "rows", "paragraphs", "tables", "old_text", "new_text", "params", "filters", "aggregations"}
    summary = {}
    for key, value in arguments.items():
        if key in bulky_keys:
            raw = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
            summary[key] = {
                "omitted_from_journal": True,
                "type": type(value).__name__,
                "serialized_chars": len(raw),
                "sha256": hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest(),
            }
        else:
            summary[key] = value
    encoded = json.dumps(summary, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(encoded) > 12000:
        return json.dumps({
            "truncated": True,
            "preview": encoded[:10000],
            "serialized_chars": len(encoded),
            "sha256": hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest(),
        }, ensure_ascii=False, separators=(",", ":"))
    return encoded


def create_tool_operation(
    operation_id: str,
    task_id: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> dict:
    """Register a tool intent and a bounded, useful summary of its arguments before dispatch."""
    if not operation_id or not task_id or not tool_name:
        raise ValueError("operation_id, task_id и tool_name обязательны")

    ts = now()
    arguments_json = _tool_arguments_summary(arguments)
    with _lock, connect() as con:
        con.execute(
            """INSERT INTO tool_operations
               (operation_id, task_id, tool_name, arguments_json, status,
                result_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'running', NULL, ?, ?)""",
            (operation_id, task_id, tool_name, arguments_json, ts, ts),
        )

    operation = get_tool_operation(operation_id)
    if operation is None:
        raise RuntimeError("Не удалось прочитать зарегистрированную операцию")
    return operation


def get_tool_operation(operation_id: str) -> dict | None:
    """Получить состояние операции по её уникальному идентификатору."""
    with connect() as con:
        row = con.execute(
            "SELECT * FROM tool_operations WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        return dict(row) if row else None


def complete_tool_operation(
    operation_id: str,
    result: dict[str, Any],
) -> dict:
    """Сохранить результат только для зарегистрированной running-операции."""
    if not isinstance(result, dict):
        raise TypeError("result должен быть словарём")

    result_json = json.dumps(
        result,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    with _lock, connect() as con:
        cur = con.execute(
            """UPDATE tool_operations
               SET status = 'completed', result_json = ?, updated_at = ?
               WHERE operation_id = ? AND status = 'running'""",
            (result_json, now(), operation_id),
        )
        if cur.rowcount != 1:
            raise ValueError(
                f"Операция {operation_id!r} не найдена или уже завершена"
            )

    operation = get_tool_operation(operation_id)
    if operation is None:
        raise RuntimeError("Не удалось прочитать завершённую операцию")
    return operation

def list_tool_operations(task_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    """Return a bounded page of durable tool receipts for one task."""
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id is required")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 1_000_000:
        raise ValueError("offset must be between 0 and 1000000")
    with connect() as con:
        rows = con.execute(
            "SELECT operation_id, task_id, tool_name, arguments_json, status, result_json, created_at, updated_at "
            "FROM tool_operations WHERE task_id=? ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
            (task_id, limit, offset),
        ).fetchall()
        return [dict(row) for row in rows]


def count_tool_operations(task_id: str) -> int:
    with connect() as con:
        row = con.execute("SELECT COUNT(*) FROM tool_operations WHERE task_id=?", (task_id,)).fetchone()
        return int(row[0]) if row else 0


def list_tasks(cid: str | None = None) -> list[dict]:
    with connect() as con:
        if cid:
            rows = con.execute("SELECT * FROM tasks WHERE conversation_id=? ORDER BY created_at DESC", (cid,)).fetchall()
        else:
            rows = con.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 100").fetchall()
        return [dict(r) for r in rows]


def record_model_request(task_id: str | None, model: str, response: dict) -> None:
    """Persist Ollama usage statistics for one completed /api/chat request."""
    if not isinstance(response, dict):
        return
    def nonnegative_int(key):
        value = response.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    if not any(nonnegative_int(key) is not None for key in ("prompt_eval_count", "eval_count")):
        return
    with _lock, connect() as con:
        con.execute(
            "INSERT INTO model_requests(task_id,model,prompt_eval_count,eval_count,prompt_eval_duration_ns,eval_duration_ns,created_at) VALUES (?,?,?,?,?,?,?)",
            (task_id, str(model or ""), nonnegative_int("prompt_eval_count"), nonnegative_int("eval_count"),
             nonnegative_int("prompt_eval_duration"), nonnegative_int("eval_duration"), now()),
        )


def get_model_usage(task_id: str) -> dict:
    with connect() as con:
        rows = con.execute("SELECT * FROM model_requests WHERE task_id=? ORDER BY id", (task_id,)).fetchall()
    items = [dict(row) for row in rows]
    prompts = [r["prompt_eval_count"] for r in items if r["prompt_eval_count"] is not None]
    outputs = [r["eval_count"] for r in items if r["eval_count"] is not None]
    return {
        "requests": len(items), "prompt_tokens_total": sum(prompts), "generated_tokens_total": sum(outputs),
        "current_prompt_tokens": prompts[-1] if prompts else 0, "max_prompt_tokens": max(prompts, default=0),
        "requests_detail": items[-100:],
    }


_TRACE_SECRET_KEYS = {"token", "auth_token", "password", "passwd", "secret", "api_key", "private_key", "authorization", "cookie"}


def _redact_trace_value(value, key: str = ""):
    import re
    if any(marker in key.lower() for marker in _TRACE_SECRET_KEYS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact_trace_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_trace_value(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", value)
        value = re.sub(r"(?i)(password|passwd|token|secret|api[_-]?key)\s*([=:])\s*[^\s,;]+", r"\1\2[REDACTED]", value)
        return value[:200_000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:2000]


def _debug_setting_enabled(key: str, env_default: bool) -> bool:
    global _debug_setting_cache, _debug_setting_cache_at
    now_mono = time.monotonic()
    with _lock:
        if now_mono - _debug_setting_cache_at > 1.0:
            try:
                with connect() as con:
                    rows = con.execute(
                        "SELECT key,value FROM app_settings WHERE key IN ('debug_llm_trace','debug_llm_thinking')"
                    ).fetchall()
                _debug_setting_cache = {
                    str(row["key"]): str(row["value"]).strip().lower() in {"1", "true", "yes", "on"}
                    for row in rows
                }
            except sqlite3.Error:
                _debug_setting_cache = {}
            _debug_setting_cache_at = now_mono
        if key in _debug_setting_cache:
            return _debug_setting_cache[key]
    return bool(env_default)


def record_debug_trace(task_id: str | None, event_type: str, payload: Any, *, thinking: bool = False) -> str | None:
    """Persist an actual trace event only when explicitly enabled; redact secrets."""
    if not task_id or not _debug_setting_enabled("debug_llm_trace", getattr(settings, "debug_llm_trace", False)):
        return None
    if thinking and not _debug_setting_enabled("debug_llm_thinking", getattr(settings, "debug_llm_thinking", False)):
        return None
    event_id = uuid.uuid4().hex
    safe_payload = _redact_trace_value(payload)
    with _lock, connect() as con:
        con.execute(
            "INSERT INTO debug_trace(event_id,task_id,event_type,payload_json,created_at) VALUES (?,?,?,?,?)",
            (event_id, task_id, str(event_type)[:80], json.dumps(safe_payload, ensure_ascii=False, default=str), now()),
        )
    return event_id


def list_debug_trace(task_id: str, limit: int = 500) -> list[dict]:
    limit = max(1, min(int(limit), 2000))
    with connect() as con:
        rows = con.execute(
            "SELECT event_id,event_type,payload_json,created_at FROM debug_trace WHERE task_id=? ORDER BY rowid LIMIT ?",
            (task_id, limit),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item.pop("payload_json"))
        except (TypeError, json.JSONDecodeError):
            item["payload"] = {"unavailable": True}
        result.append(item)
    return result


def get_app_setting(key: str, default=None):
    with connect() as con:
        con.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)")
        row = con.execute("SELECT value FROM app_settings WHERE key=?", (str(key),)).fetchone()
        return row[0] if row else default

def set_app_setting(key: str, value: str):
    global _debug_setting_cache_at
    with _lock, connect() as con:
        con.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)")
        con.execute("INSERT INTO app_settings(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at", (str(key), str(value), now()))
    if key in {"debug_llm_trace", "debug_llm_thinking"}:
        _debug_setting_cache_at = 0.0


def delete_app_setting(key: str) -> None:
    global _debug_setting_cache_at
    with _lock, connect() as con:
        con.execute("DELETE FROM app_settings WHERE key=?", (str(key),))
    if key in {"debug_llm_trace", "debug_llm_thinking"}:
        _debug_setting_cache_at = 0.0
