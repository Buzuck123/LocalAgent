from pathlib import Path
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from . import db
from .config import settings, PROJECT_ROOT
from .core import agent, initialize
from .version import VERSION
from .attachments import save_attachments, model_message_content, cleanup_conversation_attachments
from .sandbox_manager import sandbox_manager, SandboxError

app = FastAPI(title="LocalAgent", version=VERSION)
STATIC = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

class ConversationCreate(BaseModel):
    title: str = Field(default="Новый диалог", max_length=100)

class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=100)

class AttachmentInput(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    mime_type: str = Field(default="application/octet-stream", max_length=120)
    data_base64: str = Field(min_length=1, max_length=16_800_000)

class MessageCreate(BaseModel):
    content: str = Field(default="", max_length=20000)
    reasoning: bool = True
    attachments: list[AttachmentInput] = Field(default_factory=list, max_length=5)

class ModelSelect(BaseModel):
    model: str = Field(min_length=1, max_length=200)

class DebugTraceSettings(BaseModel):
    enabled: bool
    thinking: bool = False

@app.on_event("startup")
def startup():
    initialize()

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")

@app.get("/api/version")
def version():
    return {"name": "LocalAgent", "version": VERSION}

@app.get("/api/files/{relative_path:path}")
def download_published_file(relative_path: str):
    root = settings.workspace_path.resolve()
    downloads = root / "downloads"
    if downloads.is_symlink():
        raise HTTPException(404, "File not found")
    if "\x00" in relative_path or Path(relative_path).is_absolute():
        raise HTTPException(404, "File not found")
    raw_candidate = downloads / relative_path
    current = downloads
    for part in Path(relative_path).parts:
        if part in {"", ".", ".."}:
            raise HTTPException(404, "File not found")
        current = current / part
        if current.is_symlink():
            raise HTTPException(404, "File not found")
    candidate = raw_candidate.resolve()
    try:
        candidate.relative_to(downloads.resolve())
    except ValueError:
        raise HTTPException(404, "File not found")
    if not candidate.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(candidate, filename=candidate.name)


@app.get("/api/conversations")
def conversations():
    return db.list_conversations()

@app.post("/api/conversations")
def create_conversation(payload: ConversationCreate):
    return db.create_conversation(payload.title)

@app.patch("/api/conversations/{cid}")
def rename_conversation(cid: str, payload: ConversationRename):
    if not db.get_conversation(cid):
        raise HTTPException(404, "Conversation not found")
    title = payload.title.strip()
    if not title:
        raise HTTPException(400, "Название не может быть пустым")
    db.rename_conversation(cid, title)
    return db.get_conversation(cid)

@app.delete("/api/conversations/{cid}")
def delete_conversation(cid: str):
    if not db.get_conversation(cid):
        raise HTTPException(404, "Conversation not found")

    active_statuses = {"queued", "running", "paused", "cancelling"}
    active = [
        task for task in db.list_tasks(cid)
        if task["status"] in active_statuses
    ]
    if active:
        raise HTTPException(
            409,
            "В этом диалоге есть активные задачи. "
            "Дождитесь их завершения или отмените их перед удалением.",
        )

    try:
        sandbox_manager.delete(cid)
    except SandboxError as exc:
        raise HTTPException(500, f"Не удалось удалить sandbox диалога: {exc}") from exc
    try:
        attachment_cleanup = cleanup_conversation_attachments(cid)
    except OSError as exc:
        raise HTTPException(500, "Не удалось полностью удалить вложения диалога; повторите удаление.") from exc
    db.delete_conversation(cid)
    return {"ok": True, "attachments_removed": attachment_cleanup["removed"]}

@app.get("/api/conversations/{cid}/messages")
def messages(cid: str):
    if not db.get_conversation(cid):
        raise HTTPException(404, "Conversation not found")
    result = db.get_messages(cid)
    for message in result:
        message["attachments"] = [
            {key: item.get(key) for key in ("name", "mime_type", "kind", "size_bytes")}
            for item in message.get("attachments", []) if isinstance(item, dict)
        ]
    return result

@app.post("/api/conversations/{cid}/messages")
def send_message(cid: str, payload: MessageCreate):
    if not db.get_conversation(cid):
        raise HTTPException(404, "Conversation not found")
    try:
        attachments = save_attachments([item.model_dump() for item in payload.attachments])
        content = payload.content.strip()
        if not content and not attachments:
            raise ValueError("Напишите сообщение или прикрепите файл")
        display_content = content or "Проанализируй прикреплённые материалы."
        display_content += "".join(f"\n[Вложение: {item['name']}]" for item in attachments)
        task = agent.submit(cid, display_content, reasoning=payload.reasoning, attachments=attachments)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return task

def _fallback_activity(task: dict) -> str:
    status = task.get("status", "")
    return {
        "queued": "Ожидает запуска",
        "running": "Выполняется",
        "paused": "Приостановлена",
        "interrupted": "Прервана; можно продолжить",
        "completed": "Задача завершена",
        "failed": "Завершилась с ошибкой",
        "cancelled": "Задача отменена",
    }.get(status, "Состояние неизвестно")

def _parse_json_dict(value):
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _operation_status(operation):
    if operation.get("status") == "running":
        return "running", "Выполняется; результат ещё не сохранён"
    raw_result = operation.get("result_json")
    if not raw_result:
        return "unknown", "Запись не содержит сохранённого результата; выполнение не подтверждено"
    result = _parse_json_dict(raw_result)
    if not result and str(raw_result).strip() not in {"{}", "null"}:
        return "unknown", "Не удалось разобрать сохранённый результат"
    error_record = result.get("__localagent_tool_error__")
    if isinstance(error_record, dict):
        return "failed", str(error_record.get("error") or "Ошибка инструмента")[:500]
    if result.get("error") or result.get("ok") is False:
        return "failed", str(result.get("error") or "Инструмент вернул ошибку")[:500]
    if result.get("timed_out") is True:
        return "failed", "Выполнение завершилось по тайм-ауту"
    if "exit_code" in result and result.get("exit_code") not in (0, None):
        return "failed", f"Инструмент завершился с exit_code={result.get('exit_code')}"
    useful = []
    for key in ("path", "destination_path", "database_path", "table", "rows_written", "rows_imported", "rows_exported", "bytes_written", "returned_rows", "table_count", "created", "deleted"):
        if key in result and result[key] is not None:
            useful.append(f"{key}={result[key]}")
    if result.get("truncated") is True:
        useful.append("результат сокращён по лимиту вывода")
    preview = "; ".join(useful) if useful else "Операция завершена; сохранён результат инструмента"
    if str(raw_result).strip() == "null":
        return "unknown", "Результат инструмента равен null; полезный эффект не подтверждён"
    return "completed", preview[:500]


def _operation_activity(operation):
    args = _parse_json_dict(operation.get("arguments_json"))
    pieces = []
    for key in ("path", "source_path", "destination_path", "database_path", "excel_path", "table_name", "sheet_name", "action"):
        value = args.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            text = " ".join(str(value).split())
            pieces.append(f"{key}={text[:90]}")
        if len(pieces) >= 3:
            break
    return str(operation.get("tool_name") or "Инструмент") + (" · " + "; ".join(pieces) if pieces else "")


def _pause_report(task, checkpoint):
    operations = db.list_tool_operations(task["id"], limit=100)
    statuses = {"completed": [], "failed": [], "running": [], "unknown": []}
    for operation in reversed(operations):
        status, detail = _operation_status(operation)
        activity = _operation_activity(operation)
        statuses[status].append(f"{activity} — {detail}")
    reason_map = {
        "tool_call_limit": "Достигнут лимит вызовов инструментов.",
        "iteration_budget": "Исчерпан бюджет итераций.",
        "loop_detected": "Сработала защита от повторяющихся действий.",
        "consecutive_tool_failures": "Слишком много последовательных ошибок инструментов.",
        "needs_user_input": "Ожидается уточнение пользователя.",
        "completion_review_failed": "Не удалось подтвердить завершение задачи.",
    }
    reason = reason_map.get(checkpoint.get("pause_reason"), "Задача приостановлена.")
    sections = [reason]
    clarification = checkpoint.get("clarification")
    if checkpoint.get("pause_reason") == "needs_user_input" and isinstance(clarification, dict):
        question = str(clarification.get("question") or checkpoint.get("pending_question") or "").strip()
        if question and not clarification.get("resolved"):
            sections.append("Вопрос агента — ответьте обычным сообщением в этом диалоге:\n" + question)
    if statuses["completed"]:
        sections.append("Успешные операции (это подтверждает выполнение самих вызовов, не обязательно всей цели):\n- " + "\n- ".join(statuses["completed"][-8:]))
    if statuses["failed"]:
        sections.append("Ошибки:\n- " + "\n- ".join(statuses["failed"][-8:]))
    if statuses["running"]:
        sections.append("Операции, которые ещё отмечены как выполняющиеся:\n- " + "\n- ".join(statuses["running"][-5:]))
    if statuses["unknown"]:
        sections.append("Результат не подтверждён:\n- " + "\n- ".join(statuses["unknown"][-5:]))
    plan = checkpoint.get("plan")
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    if steps:
        done = [str(step.get("description", step.get("id", "этап"))) for step in steps if isinstance(step, dict) and step.get("status") == "completed"]
        remaining = [str(step.get("description", step.get("id", "этап"))) for step in steps if isinstance(step, dict) and step.get("status") != "completed"]
        sections.append("Этапы плана:\n- Завершены: " + ("; ".join(done[-10:]) if done else "нет подтверждённых") + "\n- Остались: " + ("; ".join(remaining[-10:]) if remaining else "нет"))
    sections.append("Статистика журнала: " + ", ".join(f"{name}={len(items)}" for name, items in statuses.items()) + f"; показано до {len(operations)} последних операций.")
    return "\n\n".join(sections)[:8000]


@app.get("/api/tasks/{tid}/execution")
def task_execution(tid: str):
    """Return a bounded, UI-safe summary of durable execution state and events."""
    task = db.get_task(tid)
    if not task:
        raise HTTPException(404, "Task not found")
    try:
        checkpoint = json.loads(task.get("checkpoint_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        checkpoint = {}
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    task_state = checkpoint.get("task_state")
    if not isinstance(task_state, dict):
        task_state = {}
    events = checkpoint.get("execution_events")
    if not isinstance(events, list):
        events = []
    plan = checkpoint.get("plan")
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    plan_summary = None
    if steps:
        plan_summary = {
            "goal": str(plan.get("goal") or "")[:500],
            "status": str(plan.get("status") or "unknown"),
            "total_steps": len(steps),
            "completed_steps": sum(1 for step in steps if isinstance(step, dict) and step.get("status") == "completed"),
            "remaining_steps": [str(step.get("description") or step.get("id") or "этап")[:200]
                                 for step in steps if isinstance(step, dict) and step.get("status") != "completed"][:12],
        }
    return {
        "task_id": tid,
        "phase": str(checkpoint.get("phase") or "unknown"),
        "activity": str(checkpoint.get("activity") or "")[:320],
        "task_state": {
            "actions_total": int(task_state.get("actions_total", 0) or 0),
            "actions_succeeded": int(task_state.get("actions_succeeded", 0) or 0),
            "actions_failed": int(task_state.get("actions_failed", 0) or 0),
            "successful_tools": list(task_state.get("successful_tools", []))[:50],
            "failed_categories": task_state.get("failed_categories", {}) if isinstance(task_state.get("failed_categories"), dict) else {},
            "last_error": task_state.get("last_error"),
            "unresolved_errors": list(task_state.get("unresolved_errors", []))[-10:],
        },
        "events": events[-30:],
        "plan": plan_summary,
    }


@app.get("/api/tasks/{tid}/operations")
def task_operations(tid: str, limit: int = 50, offset: int = 0):
    task = db.get_task(tid)
    if not task:
        raise HTTPException(404, "Task not found")
    if not 1 <= limit <= 200 or not 0 <= offset <= 1_000_000:
        raise HTTPException(400, "limit must be 1..200 and offset 0..1000000")
    records = db.list_tool_operations(tid, limit=limit, offset=offset)
    items = []
    total_chars = 0
    for record in records:
        status, detail = _operation_status(record)
        item = {
            "operation_id": record.get("operation_id"), "tool_name": record.get("tool_name"),
            "status": status, "status_detail": detail[:500], "activity": _operation_activity(record),
            "created_at": record.get("created_at"), "updated_at": record.get("updated_at"),
        }
        total_chars += sum(len(str(value)) for value in item.values())
        if total_chars > 30_000:
            break
        items.append(item)
    total = db.count_tool_operations(tid)
    return {"task_id": tid, "operations": items, "total": total, "offset": offset,
            "has_more": offset + len(items) < total, "note": "Tool results are summarized to keep the UI responsive."}


@app.get("/api/tasks")
def tasks(conversation_id: str | None = None):
    result = []
    for task in db.list_tasks(conversation_id):
        item = dict(task)
        try:
            checkpoint = json.loads(item.get("checkpoint_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            checkpoint = {}
        if not isinstance(checkpoint, dict):
            checkpoint = {}
        item["activity"] = checkpoint.get("activity") or _fallback_activity(item)
        item["reasoning"] = bool(checkpoint.get("reasoning", True))
        item.pop("attachments_json", None)
        item.pop("attachments", None)
        if item.get("status") == "paused":
            item["pause_report"] = _pause_report(item, checkpoint)
        # A checkpoint can contain a large transcript archive; the UI needs
        # only status/activity, not the serialized recovery state.
        item.pop("checkpoint_json", None)
        result.append(item)
    return result

def task_action(tid: str, action):
    try:
        result = action(tid)
        if isinstance(result, dict):
            result = dict(result)
            result.pop("attachments_json", None)
            result.pop("attachments", None)
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/tasks/{tid}/usage")
def task_usage(tid: str):
    task = db.get_task(tid)
    if not task:
        raise HTTPException(404, "Task not found")
    usage = db.get_model_usage(tid)
    try:
        checkpoint = json.loads(task.get("checkpoint_json") or "{}")
    except Exception:
        checkpoint = {}
    context_limit = int(checkpoint.get("model_num_ctx") or settings.num_ctx) if isinstance(checkpoint, dict) else settings.num_ctx
    usage["context_limit"] = context_limit
    usage["model"] = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    usage["current_context_percent"] = round(100 * usage["current_prompt_tokens"] / context_limit, 1) if context_limit else 0
    usage["max_context_percent"] = round(100 * usage["max_prompt_tokens"] / context_limit, 1) if context_limit else 0
    return usage

@app.post("/api/tasks/{tid}/pause")
def pause_task(tid: str):
    return task_action(tid, agent.pause)

@app.post("/api/tasks/{tid}/resume")
def resume_task(tid: str):
    return task_action(tid, agent.resume)

@app.post("/api/tasks/{tid}/cancel")
def cancel_task(tid: str):
    return task_action(tid, agent.cancel)


@app.get("/api/settings/debug-trace")
def get_debug_trace_settings():
    def resolve(key, env_default):
        stored = db.get_app_setting(key)
        if stored is None:
            return bool(env_default), "env"
        return str(stored).lower() in {"1", "true", "yes", "on"}, "ui"
    enabled, source = resolve("debug_llm_trace", settings.debug_llm_trace)
    thinking, thinking_source = resolve("debug_llm_thinking", settings.debug_llm_thinking)
    return {"enabled": enabled, "thinking": thinking, "source": source,
            "thinking_source": thinking_source}


@app.put("/api/settings/debug-trace")
def update_debug_trace_settings(payload: DebugTraceSettings):
    # UI overrides environment defaults; trace data remains disabled by default.
    db.set_app_setting("debug_llm_trace", "true" if payload.enabled else "false")
    db.set_app_setting("debug_llm_thinking", "true" if payload.enabled and payload.thinking else "false")
    return {"enabled": payload.enabled, "thinking": bool(payload.enabled and payload.thinking), "source": "ui"}


@app.get("/api/tasks/{tid}/trace")
def export_task_trace(tid: str):
    if not db.get_task(tid):
        raise HTTPException(404, "Task not found")
    return {"task_id": tid, "events": db.list_debug_trace(tid, limit=2000)}


@app.get("/api/sandbox/proxy-status")
def sandbox_proxy_status():
    """Return the last shared-proxy lifecycle check without exposing credentials."""
    return sandbox_manager.proxy_status()


@app.get("/api/ollama/model-profile")
def ollama_model_profile():
    from .core import client
    model = db.get_app_setting("selected_ollama_model", settings.model)
    try:
        return client.model_profile(model)
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc

@app.get("/api/ollama/generation")
def ollama_generation():
    from .core import client
    return dict(getattr(client, "last_generation_stats", {}))


@app.get("/api/ollama/status")
def ollama_status():
    from .core import client
    try:
        return client.status()
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc

@app.get("/api/ollama/models")
def ollama_models():
    from .core import client
    try:
        return client.models()
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc

@app.post("/api/ollama/model")
def ollama_select_model(payload: ModelSelect):
    from .core import client
    try:
        return client.select_model(payload.model)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc

@app.post("/api/ollama/unload")
def ollama_unload():
    from .core import client
    try:
        return client.unload()
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc

def main():
    # Deliberately bind to localhost by default. Do not expose to LAN without auth.
    uvicorn.run("agent.web:app", host=settings.host, port=settings.port, reload=False)

if __name__ == "__main__":
    main()
