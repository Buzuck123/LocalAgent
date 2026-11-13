import base64
from pathlib import Path
from types import SimpleNamespace

from agent import core, tools, attachments


def test_tool_registry_exposes_current_tools_and_excludes_retired_specialized_apis():
    names = {item["function"]["name"] for item in core.tools_for_task("Create a small text file")}
    assert names == {item["function"]["name"] for item in tools.TOOL_DEFINITIONS}
    assert "terminal" in names and "publish_file" in names
    assert "load_file_for_model" in names
    retired = {
        "sqlite_create_database", "sqlite_read", "sqlite_query", "sqlite_execute",
        "sqlite_import_excel", "sqlite_export_excel", "excel_aggregate",
        "read_excel", "write_excel", "read_word", "write_word",
        "copy_file", "run_python",
    }
    assert not retired & names


def test_terminal_description_exposes_persistent_linux_computer():
    definition = next(x["function"] for x in tools.TOOL_DEFINITIONS if x["function"]["name"] == "terminal")
    description = definition["description"].lower()
    assert "persistent" in description and "bash" in description and "linux" in description


def test_ollama_audio_attachment_is_sent_as_multimodal_payload(tmp_path, monkeypatch):
    audio = tmp_path / "clip.wav"
    raw = b"RIFF" + b"test-audio-bytes"
    audio.write_bytes(raw)
    monkeypatch.setattr(core, "ATTACHMENT_ROOT", tmp_path)
    monkeypatch.setattr(
        core.db, "get_app_setting",
        lambda key, default=None: "gemma4:12b" if key == "selected_ollama_model" else default,
    )
    client = core.OllamaClient()
    captured = {}

    def fake_post(endpoint, payload, timeout=None):
        captured.update(payload)
        return {"model": "gemma4:12b", "message": {"role": "assistant", "content": "heard"}, "eval_count": 20, "eval_duration": 2_000_000_000}

    monkeypatch.setattr(client, "_post", fake_post)
    client.chat([{"role": "user", "content": "Describe the audio", "attachments": [{"kind": "audio", "name": "clip.wav", "path": str(audio), "source": "managed_attachment"}]}])
    assert captured["messages"][0]["images"] == [base64.b64encode(raw).decode("ascii")]
    assert client.last_generation_stats["tokens_per_second"] == 10.0
    assert client.last_generation_stats["model"] == "gemma4:12b"


def test_generation_speed_is_available_through_api(monkeypatch):
    from agent import web
    class FakeClient:
        last_generation_stats = {"tokens_per_second": 42.5, "eval_count": 85, "eval_duration": 2_000_000_000, "model": "test-model"}
    monkeypatch.setattr("agent.core.client", FakeClient())
    result = web.ollama_generation()
    assert result["tokens_per_second"] == 42.5


def test_ollama_model_selection_updates_client_state(monkeypatch):
    client = core.OllamaClient()
    monkeypatch.setattr(client, "models", lambda: {"models": [{"name": "model-b"}]})
    monkeypatch.setattr(core.db, "set_app_setting", lambda key, value: None)
    result = client.select_model("model-b")
    assert result["selected_model"] == "model-b"
    assert client.model == "model-b"


def test_simple_single_file_task_skips_planner():
    assert core.classify_execution_mode("Создай файл с точным текстом") == "simple"


def test_invalid_planner_json_falls_back_to_direct_execution(monkeypatch):
    calls = []
    monkeypatch.setattr(core, "_client_chat", lambda *a, **k: (calls.append(1) or {"message": {"content": "not json"}}))
    plan = core._create_plan_with_retries("Develop a small but complete application", max_attempts=2)
    assert len(calls) == 2
    assert plan["planner_fallback"]["used"] is True
    assert plan["steps"][0]["id"] == "fallback_direct_execution"


def test_json_parser_extracts_embedded_json_and_trailing_comma():
    parsed = core._parse_json_object('Here is the result:\n{"ok": true, "items": [1, 2,],}\nDone')
    assert parsed == {"ok": True, "items": [1, 2]}


def test_uploaded_audio_is_normalized_for_model_input(tmp_path, monkeypatch):
    calls = []
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(attachments, "settings", SimpleNamespace(database_path=data_dir / "agent.sqlite3"))
    monkeypatch.setattr(attachments, "ATTACHMENT_ROOT", data_dir / "attachments")
    monkeypatch.setattr(attachments.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        calls.append(command)
        if "ffprobe" in command[0]:
            return SimpleNamespace(returncode=0, stdout="12.5", stderr="")
        Path(command[-1]).write_bytes(b"RIFF" + b"normalized-wav")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(attachments.subprocess, "run", fake_run)
    item = {"name": "voice.mp3", "mime_type": "audio/mpeg", "data_base64": base64.b64encode(b"fake-mp3").decode("ascii")}
    saved = attachments.save_attachments([item])[0]
    assert saved["kind"] == "audio"
    assert saved["path"].endswith(".wav")
    assert Path(saved["path"]).read_bytes().startswith(b"RIFF")
    assert saved["duration_seconds"] == 12.5
    assert "-ar" in calls[1] and "16000" in calls[1]
