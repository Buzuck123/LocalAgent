import pytest

from agent import core, tools


def test_ask_model_uses_clean_context_fixed_model_and_no_tools(monkeypatch):
    calls = []

    def fake_chat(messages, tools=None, reasoning=True, model=None):
        calls.append((messages, tools, reasoning, model))
        return {"model": model, "message": {"content": "isolated answer"}}

    monkeypatch.setattr(core, "_client_chat", fake_chat)
    result = core._ask_model_tool(
        {"prompt": "Analyze this independently"},
        conversation_id="conversation-1",
        context_messages=[{"role": "user", "content": "PRIVATE MAIN TASK CONTEXT"}],
        main_model="qwen3.5:9b",
        context_limit=32768,
    )

    assert result["ok"] is True
    assert result["answer"] == "isolated answer"
    assert len(calls) == 1
    messages, tool_schemas, reasoning, model = calls[0]
    assert tool_schemas is None
    assert model == "gemma4:12b-64k-32k-42gpu"
    assert reasoning is True
    assert len(messages) == 2
    assert "PRIVATE MAIN TASK CONTEXT" not in str(messages)
    assert messages[1]["content"] == "Analyze this independently"


def test_ask_model_requires_nonempty_prompt():
    with pytest.raises(ValueError, match="prompt"):
        core._ask_model_tool(
            {"prompt": "  "},
            conversation_id="conversation-1",
            context_messages=[],
            main_model="qwen3.5:9b",
            context_limit=32768,
        )


def test_ask_model_attaches_supported_container_text_file(monkeypatch):
    loaded_paths = []
    calls = []

    def fake_load(path, *, conversation_id):
        loaded_paths.append((path, conversation_id))
        return {
            "ok": True, "kind": "text", "name": "notes.txt",
            "extracted_text": "file payload", "size_bytes": 12,
        }

    def fake_chat(messages, tools=None, reasoning=True, model=None):
        calls.append(messages)
        return {"message": {"content": "reviewed"}}

    monkeypatch.setattr(tools, "load_file_for_model", fake_load)
    monkeypatch.setattr(core, "_client_chat", fake_chat)
    result = core._ask_model_tool(
        {"prompt": "Review", "file": "/etc/notes.txt"},
        conversation_id="conversation-2",
        context_messages=[],
        main_model="qwen3.5:9b",
        context_limit=32768,
    )

    assert result["answer"] == "reviewed"
    assert loaded_paths == [("/etc/notes.txt", "conversation-2")]
    assert "file payload" in calls[0][1]["content"]


def test_ask_model_rejects_unsupported_file_type_before_download(monkeypatch):
    def should_not_load(*args, **kwargs):
        raise AssertionError("unsupported file must be rejected before download")

    monkeypatch.setattr(tools, "load_file_for_model", should_not_load)
    with pytest.raises(ValueError, match="Формат файла"):
        core._ask_model_tool(
            {"prompt": "Review", "file": "/tmp/archive.bin"},
            conversation_id="conversation-3",
            context_messages=[],
            main_model="qwen3.5:9b",
            context_limit=32768,
        )


def test_ask_model_reports_gemma_context_overflow(monkeypatch):
    def overflow(*args, **kwargs):
        raise RuntimeError("prompt exceeds context length")

    monkeypatch.setattr(core, "_client_chat", overflow)
    result = core._ask_model_tool(
        {"prompt": "Analyze"},
        conversation_id="conversation-4",
        context_messages=[],
        main_model="qwen3.5:9b",
        context_limit=32768,
    )

    assert result["ok"] is True
    assert result["context_overflow"] is True
    assert "переполнил контекст" in result["warning"]
    assert "Ответ модели не получен" in result["answer"]


def test_ask_model_refuses_when_main_context_has_no_safe_output_budget():
    result = core._ask_model_tool(
        {"prompt": "Analyze"},
        conversation_id="conversation-5",
        context_messages=[{"role": "user", "content": "x" * 50000}],
        main_model="gemma4:12b",
        context_limit=2048,
    )
    assert result["ok"] is False
    assert "Недостаточно свободного контекста" in result["error"]


def test_ask_model_is_advertised_with_only_prompt_and_optional_file():
    schema = next(
        item["function"]
        for item in tools.TOOL_DEFINITIONS
        if item["function"]["name"] == "ask_model"
    )
    assert schema["parameters"]["required"] == ["prompt"]
    assert set(schema["parameters"]["properties"]) == {"prompt", "file"}
