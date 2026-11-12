"""Bounded working-context management for long-running agent tasks.

The checkpoint keeps the archived transcript for diagnostics; only the compacted
working message list is sent to the LLM. Compaction is called at safe loop
boundaries, never while a tool call is awaiting its result.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable

CONTEXT_SUMMARY_MARKER = "[LOCAL_AGENT_CONTEXT_SUMMARY]"
PLAN_CONTEXT_MARKER = "[LOCAL_AGENT_PLAN_CONTEXT]"


def ensure_event_ids(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign stable IDs to non-system events; never infer duplicates from text."""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        if not isinstance(message.get("_event_id"), str) or not message.get("_event_id"):
            message["_event_id"] = uuid.uuid4().hex
    return messages


def deduplicate_event_ids(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove only repeated copies carrying the exact same event ID."""
    seen: set[str] = set()
    result = []
    removed = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") == "system":
            result.append(message)
            continue
        event_id = message.get("_event_id")
        if isinstance(event_id, str) and event_id:
            if event_id in seen:
                removed += 1
                continue
            seen.add(event_id)
        result.append(message)
    return result, removed


def estimate_tokens(messages: list[dict[str, Any]], model: str | None = None,
                    tokenizer: Callable[[str], int] | None = None) -> int:
    """Estimate model context size; use an injected tokenizer when available.

    Ollama does not expose a universal tokenizer endpoint for arbitrary tags.
    The fallback is conservative and family-aware, with a CJK adjustment rather
    than assuming every character costs exactly one third of a token.
    """
    text_parts = []
    overhead = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        text_parts.append(str(message.get("content", "")))
        text_parts.append(str(message.get("tool_calls", "")))
        overhead += 32
    text = "\n".join(text_parts)
    if tokenizer is not None:
        try:
            return max(1, int(tokenizer(text)) + overhead)
        except Exception:
            pass
    try:
        import tiktoken
        if model:
            try:
                encoding = tiktoken.encoding_for_model(model)
                return max(1, len(encoding.encode(text)) + overhead)
            except Exception:
                pass
    except ImportError:
        pass
    model_name = str(model or "").lower()
    chars_per_token = 3.0 if any(name in model_name for name in ("qwen", "gemma", "deepseek")) else 3.3
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" or "\uac00" <= ch <= "\ud7af")
    other = max(0, len(text) - cjk)
    return max(1, int(other / chars_per_token + cjk / 1.5) + overhead)


def context_growth_forecast(messages: list[dict[str, Any]], num_ctx: int, *,
                            expected_output_tokens: int | None = None,
                            model: str | None = None) -> dict[str, Any]:
    """Return current and projected context usage without mutating history."""
    limit = max(1, int(num_ctx or 1))
    current = estimate_tokens(messages, model=model)
    recent_tool_chars = [
        len(str(m.get("content", ""))) for m in messages[-8:]
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    observed_growth = int(sum(recent_tool_chars[-3:]) / 3 / 3) if recent_tool_chars else 0
    response_reserve = max(0, int(expected_output_tokens or 0))
    expected = response_reserve + observed_growth
    projected = current + expected
    return {
        "model": model,
        "estimated_tokens": current,
        "context_limit": limit,
        "usage_ratio": round(current / limit, 4),
        "observed_next_input_tokens": observed_growth,
        "response_reserve_tokens": response_reserve,
        "expected_growth_tokens": expected,
        "projected_tokens": projected,
        "projected_ratio": round(projected / limit, 4),
        "estimate_kind": "conservative_heuristic",
    }


def needs_compaction(messages: list[dict[str, Any]], num_ctx: int, threshold: float = 0.70,
                     model: str | None = None) -> bool:
    try:
        budget = max(2048, int(num_ctx) * threshold)
    except (TypeError, ValueError):
        budget = 19660
    return estimate_tokens(messages, model=model) >= budget


def _short(value: Any, limit: int = 500) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    value = " ".join(value.split())
    if len(value) > limit:
        return value[: limit - 30] + " …[сокращено]… " + value[-15:]
    return value


def _message_summary(message: dict[str, Any]) -> list[str]:
    role = message.get("role")
    if role == "system":
        return []
    if role == "user":
        rows = ["Сообщение пользователя (сохранено как факт диалога): " + _short(message.get("content", ""), 1600)]
        attachments = message.get("attachments", [])
        if isinstance(attachments, list) and attachments:
            refs = []
            for item in attachments[:8]:
                if not isinstance(item, dict):
                    continue
                refs.append({key: item.get(key) for key in ("name", "kind", "path", "source") if item.get(key) is not None})
            if refs:
                rows.append("Вложения пользовательского сообщения: " + _short(refs, 600))
        return rows
    if role == "assistant" and message.get("tool_calls"):
        rows = []
        for call in message.get("tool_calls", [])[:4]:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            rows.append("Вызов инструмента: " + _short({
                "name": fn.get("name"), "arguments": fn.get("arguments")
            }, 350))
        return rows
    if role == "tool":
        internal = message.get("_localagent_internal")
        internal_rows = []
        if isinstance(internal, dict):
            private_facts = {
                key: internal.get(key) for key in
                ("_terminal_command", "result_id", "state", "timed_out", "interrupted",
                 "duration_seconds", "shell_state", "output_truncated")
                if internal.get(key) is not None
            }
            if private_facts:
                internal_rows.append(
                    f"Служебное состояние terminal (не часть экранного вывода): {_short(private_facts, 500)}"
                )
        raw = str(message.get("content", ""))
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            parsed = raw
        if isinstance(parsed, dict):
            result_id = parsed.get("result_id")
            ref = f" [result_id={result_id}]" if result_id else ""
            error = parsed.get("error")
            if error:
                return internal_rows + [f"Ошибка инструмента {message.get('tool_name', '?')}{ref}: {_short(error, 400)}"]
            result = parsed.get("result", parsed)
            if isinstance(result, dict):
                useful = {k: result[k] for k in (
                    "path", "source", "destination", "deleted", "exit_code", "timed_out",
                    "stdout", "stderr", "stdout_artifact", "stderr_artifact",
                    "stdout_truncated", "stderr_truncated", "stdout_capture_truncated",
                    "stderr_capture_truncated", "stdout_total_bytes", "stderr_total_bytes",
                    "output_dir", "error", "entries", "content",
                    "bytes_written", "acceptance_proof", "agent_diagnosis"
                ) if k in result}
                if not useful:
                    useful = result
                return internal_rows + [f"Результат {message.get('tool_name', '?')}{ref}: {_short(useful, 500)}"]
            return internal_rows + [f"Результат {message.get('tool_name', '?')}{ref}: {_short(parsed, 400)}"]
        return internal_rows + [f"Результат {message.get('tool_name', '?')}: {_short(parsed, 400)}"]
    # Free-form assistant prose is deliberately not promoted into durable facts or
    # instructions. The current plan and controller state are re-injected separately;
    # tool calls/results and user messages are the reliable execution record.
    return []


def _clip_recent_tool_output(messages: list[dict[str, Any]], max_chars: int = 4200) -> list[dict[str, Any]]:
    clipped = []
    for original in messages:
        message = dict(original)
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str) and len(content) > max_chars:
            message["content"] = (
                content[: max_chars - 900]
                + "\n...[результат сокращён для рабочего контекста; полная версия сохранена в архиве контрольной точки]...\n"
                + content[-800:]
            )
        clipped.append(message)
    return clipped


def compact_messages(
    messages: list[dict[str, Any]],
    *,
    goal: str,
    previous_summary: str = "",
    keep_recent: int = 12,
    max_summary_chars: int = 24000,
    prior_archived_messages: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    """Return working messages, a factual summary, and newly archived messages.

    Prior summaries are intentionally not carried forward: older versions allowed an
    LLM-written summary (including speculative next_actions) to become input to the
    next summary, turning a lossy paraphrase into persistent instructions. The raw
    archived transcript is retained in the checkpoint and is the source for rebuilding
    the compact factual projection.
    """
    system_messages = [
        dict(m) for m in messages
        if isinstance(m, dict) and m.get("role") == "system"
        and not str(m.get("content", "")).startswith((CONTEXT_SUMMARY_MARKER, PLAN_CONTEXT_MARKER))
    ]
    non_system = [dict(m) for m in messages if isinstance(m, dict) and m.get("role") != "system"]
    cut = max(0, len(non_system) - keep_recent)
    # Never start the retained tail with an orphan tool reply. Keep its assistant
    # tool-call message too, so the LLM sees a valid call/result pair.
    while cut > 0 and non_system[cut].get("role") == "tool":
        cut -= 1
    if cut > 0 and non_system[cut - 1].get("role") == "assistant" and non_system[cut - 1].get("tool_calls"):
        cut -= 1

    archived = non_system[:cut]
    recent = non_system[cut:]
    # Rebuild from the durable raw archive rather than recursively summarizing a
    # previous LLM-authored summary. Keep user messages, tool calls and observations;
    # omit assistant prose because it may contain speculative or stale next actions.
    prior = [dict(m) for m in (prior_archived_messages or []) if isinstance(m, dict)]
    factual_history = prior + archived
    goal_line = "Текущая задача пользователя (оригинальный запрос; authoritative):\n" + str(goal or "")[:20000]
    history_lines: list[str] = []
    # Bound the projection while retaining a useful window of actual interactions.
    for message in factual_history[-120:]:
        history_lines.extend(_message_summary(message))
    # Prefer recent evidence; remove adjacent duplicates without reordering.
    deduped = []
    for line in history_lines:
        if line and (not deduped or line != deduped[-1]):
            deduped.append(line)
    summary_parts = [
        "[LOCAL_AGENT_CONTEXT_SUMMARY_V2]",
        goal_line,
        "Фактическая история (пользовательские сообщения, вызовы и результаты инструментов; не является новым планом):",
        *deduped,
    ]
    summary = "\n".join(summary_parts)
    if len(summary) > max_summary_chars:
        # Keep the original request intact and trim the oldest factual observations.
        header = "\n".join(summary_parts[:3])
        available = max(0, max_summary_chars - len(header) - 32)
        tail = "\n".join(deduped)[-available:] if available else ""
        summary = header + ("\n…[ранние наблюдения опущены]…\n" + tail if tail else "")

    # Preserve full versions of oversized recent messages in the checkpoint archive.
    clipped_recent = _clip_recent_tool_output(recent)
    for original, clipped in zip(recent, clipped_recent):
        if original != clipped:
            archived.append(original)
            summary += "\n" + "\n".join(_message_summary(original))
    if len(summary) > max_summary_chars:
        header = "\n".join(summary.splitlines()[:3])
        available = max(0, max_summary_chars - len(header) - 32)
        summary = header + ("\n…[сводка сокращена]…\n" + summary[-available:] if available else "")

    summary_message = {
        "role": "system",
        "content": CONTEXT_SUMMARY_MARKER + "\n" + summary
    }
    if system_messages:
        working = [system_messages[0], summary_message, *system_messages[1:], *clipped_recent]
    else:
        working = [summary_message, *clipped_recent]
    return working, summary, archived


def retrieve_tool_result_fragment(archive: dict[str, str], result_id: str, start: int = 0, length: int = 4000) -> dict[str, Any]:
    """Return a bounded fragment from an archived raw tool result."""
    if not isinstance(result_id, str) or not result_id.strip():
        return {"ok": False, "error": "result_id must be a non-empty string"}
    if not isinstance(archive, dict) or result_id not in archive:
        return {"ok": False, "error": f"Archived tool result not found: {result_id}"}
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        return {"ok": False, "error": "start must be a non-negative integer"}
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        return {"ok": False, "error": "length must be a positive integer"}
    # The caller controls chunk size; preserve exact offset semantics without a hidden cap.
    raw = archive[result_id]
    if not isinstance(raw, str):
        raw = json.dumps(raw, ensure_ascii=False, default=str)
    start = min(start, len(raw))
    end = min(len(raw), start + length)
    return {
        "ok": True,
        "result_id": result_id,
        "start": start,
        "end": end,
        "total_chars": len(raw),
        "has_more": end < len(raw),
        "next_start": end if end < len(raw) else None,
        "content": raw[start:end],
    }
