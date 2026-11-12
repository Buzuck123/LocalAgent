import json

from agent.context import (compact_messages, estimate_tokens, needs_compaction, retrieve_tool_result_fragment,
                           ensure_event_ids, deduplicate_event_ids, context_growth_forecast)


def test_context_compaction_keeps_system_goal_and_recent_tool_pair():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.txt"}}}]},
        {"role": "tool", "tool_name": "read_file", "content": json.dumps({"ok": True, "result": {"content": "x"}})},
        {"role": "user", "content": "current question"},
        {"role": "assistant", "content": "current answer"},
    ]
    compacted, summary, archived = compact_messages(
        messages, goal="current task", prior_archived_messages=[
            {"role": "tool", "tool_name": "list_files", "content": json.dumps({"ok": True, "result": {"entries": ["prior.txt"]}})}
        ], previous_summary="stale next_actions: build the whole project now", keep_recent=3
    )
    assert compacted[0]["content"] == "system prompt"
    assert any("current task" in m.get("content", "") for m in compacted if m.get("role") == "system")
    assert "prior.txt" in summary
    assert "stale next_actions" not in summary
    assert "LOCAL_AGENT_CONTEXT_SUMMARY_V2" in summary
    assert archived
    assert compacted[-1]["content"] == "current answer"
    assert not (compacted[-1].get("role") == "tool")


def test_context_estimator_triggers_before_full_context():
    messages = [{"role": "user", "content": "x" * 80000}]
    assert estimate_tokens(messages) > 10000
    assert needs_compaction(messages, 32768)


def test_context_summary_keeps_tool_result_retrieval_id():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "tool", "tool_name": "run_python", "content": json.dumps({
            "ok": False, "result_id": "tr_abc123", "result": {"exit_code": 1, "stderr": "Traceback"}
        })},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "next"},
    ]
    _, summary, _ = compact_messages(messages, goal="fix script", keep_recent=1)
    assert "tr_abc123" in summary


def test_retrieve_tool_result_fragment_is_bounded_and_reports_remaining_data():
    archive = {"tr_demo": "0123456789" * 1000}
    result = retrieve_tool_result_fragment(archive, "tr_demo", start=10, length=99999)
    assert result["ok"] is True
    assert result["start"] == 10
    assert len(result["content"]) == 6000
    assert result["has_more"] is True
    assert result["total_chars"] == 10000


def test_retrieve_tool_result_fragment_rejects_unknown_id():
    result = retrieve_tool_result_fragment({}, "tr_missing")
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_context_compaction_does_not_promote_assistant_prose_to_instructions():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "assistant", "content": "Remaining work: build every report immediately."},
        {"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": {"path": "stage1.txt"}}}]},
        {"role": "tool", "tool_name": "write_file", "content": json.dumps({"ok": True, "result": {"path": "stage1.txt", "bytes_written": 10}})},
        {"role": "user", "content": "Do not overwrite the existing source file."},
        {"role": "assistant", "content": "Current answer"},
    ]
    compacted, summary, archived = compact_messages(messages, goal="Prepare project", keep_recent=2)
    assert "Remaining work: build every report immediately" not in summary
    assert "stage1.txt" in summary
    assert any(m.get("role") == "user" and "Do not overwrite the existing source file" in m.get("content", "") for m in compacted)
    assert archived


def test_compaction_rebuilds_from_raw_archive_across_multiple_compactions():
    prior = [
        {"role": "assistant", "content": "Do all remaining stages now."},
        {"role": "tool", "tool_name": "write_file", "content": json.dumps({"ok": True, "result": {"path": "verified.csv"}})},
    ]
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "tool", "tool_name": "read_file", "content": json.dumps({"ok": True, "result": {"path": "verified.csv", "content": "sku,qty"}})},
        {"role": "user", "content": "Keep the SKU column unchanged."},
        {"role": "assistant", "content": "Continuing."},
    ]
    _, summary, _ = compact_messages(
        messages, goal="Create inventory", prior_archived_messages=prior, keep_recent=1
    )
    assert "verified.csv" in summary
    assert "Keep the SKU column unchanged" in summary
    assert "Do all remaining stages now" not in summary


def test_event_deduplication_uses_ids_not_equal_text():
    messages = [
        {"role": "user", "content": "same"},
        {"role": "user", "content": "same"},
    ]
    ensure_event_ids(messages)
    assert messages[0]["_event_id"] != messages[1]["_event_id"]
    duplicate = dict(messages[0])
    compacted, removed = deduplicate_event_ids([*messages, duplicate])
    assert removed == 1
    assert len(compacted) == 2


def test_context_growth_forecast_includes_model_and_expected_output():
    messages = [{"role": "user", "content": "x" * 1000}]
    stats = context_growth_forecast(messages, 10000, expected_output_tokens=2000, model="qwen3.5:9b")
    assert stats["model"] == "qwen3.5:9b"
    assert stats["expected_growth_tokens"] == 2000
    assert stats["projected_tokens"] > stats["estimated_tokens"]
    assert stats["usage_ratio"] < stats["projected_ratio"]


def test_llm_context_compaction_preserves_exact_goal_and_uses_model_summary(monkeypatch):
    from agent import core

    captured = {}
    def fake_chat(messages, tools=None, reasoning=True):
        captured["messages"] = messages
        return {"message": {"content": "Факты: файл app.py создан и тесты пройдены."}}
    monkeypatch.setattr(core, "_client_chat", fake_chat)
    summary = core._compact_context_summary_with_llm(
        "Создай приложение и проверь его.",
        "История: вызов terminal создал app.py; pytest завершился успешно.",
    )
    assert summary.startswith("Исходный запрос пользователя (сохранён дословно):")
    assert "Создай приложение и проверь его." in summary
    assert "Файл app.py" in summary or "файл app.py" in summary
    assert "Детерминированная фактическая проекция" in captured["messages"][1]["content"]


def test_llm_context_compaction_falls_back_when_model_returns_empty(monkeypatch):
    from agent import core
    monkeypatch.setattr(core, "_client_chat", lambda *a, **k: {"message": {"content": ""}})
    assert core._compact_context_summary_with_llm("goal", "facts") is None


def test_direct_file_load_is_rejected_when_it_would_consume_context_reserve():
    from agent.core import _guard_loaded_file_context

    result = _guard_loaded_file_context(
        {"ok": True, "kind": "text", "name": "large.txt",
         "extracted_text": "x" * 180000},
        [],
        {"model_num_ctx": 32768, "model_num_predict": 4096},
        "qwen3.5:9b",
    )
    assert result["ok"] is False
    assert result["file_not_loaded"] is True
    assert "контекстное окно" in result["error"].lower()
    assert "раздели файл" in result["error"].lower()


def test_direct_file_load_fits_when_context_has_room():
    from agent.core import _guard_loaded_file_context

    result = _guard_loaded_file_context(
        {"ok": True, "kind": "text", "name": "small.txt",
         "extracted_text": "Короткий текст."},
        [],
        {"model_num_ctx": 32768, "model_num_predict": 4096},
        "qwen3.5:9b",
    )
    assert result["ok"] is True


def test_actual_context_overflow_removes_direct_file_payload():
    import json
    from agent.core import _replace_overflowed_file_load_with_error

    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_name": "load_file_for_model",
         "content": json.dumps({"ok": True, "result": {"kind": "image"}})},
        {"role": "user", "content": "Attached image", "attachments": [{"kind": "image", "path": "/tmp/a.png"}]},
    ]
    assert _replace_overflowed_file_load_with_error(messages) is True
    assert len(messages) == 2
    payload = json.loads(messages[-1]["content"])
    assert payload["ok"] is False
    assert "переполнении контекстного окна" in payload["error"]


def test_context_growth_forecast_adds_observed_input_growth_and_response_reserve():
    from agent.context import context_growth_forecast

    messages = [
        {"role": "user", "content": "x"},
        {"role": "tool", "content": "z" * 9000},
        {"role": "tool", "content": "z" * 9000},
        {"role": "tool", "content": "z" * 9000},
    ]
    stats = context_growth_forecast(messages, 50000, expected_output_tokens=2000, model="qwen3.5:9b")
    assert stats["observed_next_input_tokens"] > 0
    assert stats["response_reserve_tokens"] == 2000
    assert stats["expected_growth_tokens"] == stats["observed_next_input_tokens"] + 2000
