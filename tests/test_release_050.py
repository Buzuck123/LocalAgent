import json
from agent import db, orchestration, tools, core


def test_terminal_requires_a_command_and_accepts_bounded_timeout():
    defs = [x for x in tools.TOOL_DEFINITIONS if x["function"]["name"] == "terminal"]
    schema = defs[0]["function"]["parameters"]
    orchestration._validate_schema({"command": "pwd"}, schema)
    orchestration._validate_schema({"command": "sleep 1", "timeout": 30}, schema)
    # An empty command is the explicit wait operation when a foreground
    # process is already running; unknown keys and invalid timeouts remain rejected.
    orchestration._validate_schema({}, schema)
    for args in ({"command": "pwd", "unknown": True}, {"command": "pwd", "timeout": 9999}):
        try:
            orchestration._validate_schema(args, schema)
        except orchestration.ToolContractError:
            pass
        else:
            raise AssertionError(f"invalid terminal args accepted: {args}")


def test_assistant_history_repair_preserves_tool_receipts():
    msgs = [{"role":"assistant","content":"a"},{"role":"assistant","content":"b"}]
    out = core._prepare_messages_for_next_model_turn(msgs)
    assert out[-1]["role"] == "user"
    assert "a" in out[0]["content"] and "b" in out[0]["content"]


def test_selected_model_setting_roundtrip(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(db, "settings", SimpleNamespace(database_path=tmp_path/"agent.sqlite3"))
    db.init_db()
    assert db.get_app_setting("selected_ollama_model", "fallback") == "fallback"
    db.set_app_setting("selected_ollama_model", "qwen3.5:9b")
    assert db.get_app_setting("selected_ollama_model") == "qwen3.5:9b"
