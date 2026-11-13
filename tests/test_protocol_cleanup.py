"""Regression tests for the simplified model/tool protocol."""
import pytest

from agent.tools import TOOL_DEFINITIONS
from agent.core import classify_execution_mode
from agent.orchestration import ToolContractError, validate_tool_call


def test_model_tool_catalog_contains_no_legacy_or_self_test_tools():
    names = {item["function"]["name"] for item in TOOL_DEFINITIONS}
    assert names == {
        "get_tool_result", "terminal", "publish_file", "load_file_for_model", "ask_model",
    }
    assert not names.intersection({"read_output_file", "run_python", "self_test_tools"})


def test_publish_file_schema_exposes_only_source_path():
    definition = next(item for item in TOOL_DEFINITIONS
                      if item["function"]["name"] == "publish_file")
    schema = definition["function"]["parameters"]
    assert schema["required"] == ["source_path"]
    assert set(schema["properties"]) == {"source_path"}


def test_terminal_command_max_length_is_enforced_server_side():
    with pytest.raises(ToolContractError):
        validate_tool_call("terminal", {"command": "x" * 100001}, TOOL_DEFINITIONS)


def test_stage_reviewer_receives_success_evidence_once_and_keeps_recent_errors(monkeypatch):
    import json
    from agent import core

    captured = {}
    def fake_chat(messages, tools=None, reasoning=True):
        captured["payload"] = json.loads(messages[1]["content"])
        return {"message": {"content": json.dumps({
            "decision": "continue_step",
            "reason": "The current stage needs one more check.",
            "task_prompt": "Run the missing check.",
            "reconciled_steps": [],
        })}}
    monkeypatch.setattr(core, "_client_chat", fake_chat)
    plan = {"goal": "Complete a small task", "steps": []}
    step = {"id": "step_1", "expected_result": "Result", "success_criteria": ["Result"]}
    messages = [
        {"role": "tool", "tool_name": "terminal", "content": json.dumps({"ok": True, "result": {"stdout": "verified"}})},
        {"role": "tool", "tool_name": "terminal", "content": json.dumps({"ok": False, "error": "missing file"})},
    ]
    core._review_step(
        plan, step, "The latest attempt needs another check.", messages,
        step_tool_evidence=[{"tool_name": "terminal", "result_summary": "verified"}],
        planning_context={"recent_dialogue": [{"role": "user", "content": "duplicate"}], "steps": []},
    )
    recent = captured["payload"]["recent_interaction"]
    assert len(recent) == 1
    assert recent[0]["role"] == "tool_error"
    assert "missing file" in recent[0]["error"]
    assert "recent_dialogue" not in captured["payload"]["planning_context"]


def test_pause_report_shows_question_and_explains_how_to_answer(monkeypatch):
    from agent import web

    monkeypatch.setattr(web.db, "list_tool_operations", lambda *args, **kwargs: [])
    report = web._pause_report(
        {"id": "task-1"},
        {
            "pause_reason": "needs_user_input",
            "clarification": {"question": "Which output format should I use?", "resolved": False},
        },
    )
    assert "Which output format should I use?" in report
    assert "обычным сообщением в этом диалоге" in report


def test_independent_output_count_does_not_force_plan():
    assert classify_execution_mode("Создай два независимых текстовых файла") == "simple"


def test_genuinely_multi_stage_project_uses_plan_fallback():
    assert classify_execution_mode("Разработай приложение с несколькими этапами и проверками") == "planned"


def test_independent_outputs_with_an_attachment_still_use_direct():
    assert classify_execution_mode(
        "Создай два независимых текстовых файла",
        [{"name": "source.txt", "kind": "document"}],
    ) == "simple"


def test_legacy_checkpoint_tools_are_rejected_by_the_model_contract():
    with pytest.raises(ToolContractError):
        validate_tool_call("write_file", {"path": "x.txt", "content": "x"}, TOOL_DEFINITIONS)
    with pytest.raises(ToolContractError):
        validate_tool_call("run_python", {"path": "x.py"}, TOOL_DEFINITIONS)


def test_load_file_schema_exposes_only_real_inputs_and_chunk_offsets():
    definition = next(item for item in TOOL_DEFINITIONS
                      if item["function"]["name"] == "load_file_for_model")
    properties = definition["function"]["parameters"]["properties"]
    assert set(properties) == {"path", "start", "length"}
    assert properties["start"]["minimum"] == 0
    assert properties["length"]["minimum"] == 1


def test_terminal_result_id_is_visible_but_runtime_metadata_is_not():
    import json
    from agent.core import _build_tool_messages

    raw_result = {
        "_terminal_text": "command output",
        "_localagent_internal": {"exit_code": 1, "operation_id": "internal-op-1", "pool": "pool-1"},
    }
    messages = _build_tool_messages(
        "terminal",
        json.dumps({"ok": True, "result": raw_result, "result_id": "tr_123"}),
    )
    assert "command output" in messages[0]["content"]
    assert "tr_123" in messages[0]["content"]
    assert "exit_code" not in messages[0]["content"]
    assert "internal-op-1" not in messages[0]["content"]
    assert "pool-1" not in messages[0]["content"]


def test_tool_result_retrieval_representation_hides_internal_terminal_fields():
    import json
    from agent.core import _model_visible_tool_content

    raw = json.dumps({
        "ok": True,
        "result": {
            "_terminal_text": "full output",
            "_localagent_internal": {"exit_code": 2, "operation_id": "secret"},
        },
    })
    visible = json.loads(_model_visible_tool_content("terminal", raw))
    assert visible["result"] == {"output": "full output"}
    assert "secret" not in json.dumps(visible)
    assert "exit_code" not in json.dumps(visible)


def test_terminal_archive_retrieval_keeps_full_output_not_only_screen_preview():
    from agent.core import _model_visible_tool_result

    result = _model_visible_tool_result("terminal", {
        "_terminal_text": "screen preview (truncated)",
        "_terminal_output": "complete captured terminal output",
        "_localagent_internal": {"exit_code": 0, "operation_id": "hidden"},
    })
    assert result == {"output": "complete captured terminal output"}


def test_ollama_message_boundary_filters_retired_tools_and_old_publish_arguments():
    import json
    from agent.core import _sanitize_messages_for_ollama

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "old-call", "function": {"name": "write_file", "arguments": {"path": "x", "content": "x"}}},
                {"id": "publish-call", "function": {"name": "publish_file", "arguments": {
                    "source_path": "/work/report.pdf", "destination_path": "old/path", "overwrite": True,
                }}},
            ],
        },
        {
            "role": "tool", "tool_name": "write_file",
            "content": json.dumps({"ok": True, "result": {"path": "x"}}),
        },
        {
            "role": "tool", "tool_name": "terminal_start",
            "content": json.dumps({"ok": True, "result": {"operation_id": "secret-op", "state": "running"}}),
        },
        {
            "role": "tool", "tool_name": "terminal",
            "content": json.dumps({"ok": True, "result": {
                "_terminal_text": "visible output", "_terminal_output": "visible output",
                "_localagent_internal": {"exit_code": 0, "operation_id": "secret-op"},
            }, "result_id": "tr_42"}),
        },
    ]
    safe = _sanitize_messages_for_ollama(messages)
    encoded = json.dumps(safe)
    assert "write_file" not in encoded
    assert "terminal_start" not in encoded
    assert "secret-op" not in encoded
    assert "exit_code" not in encoded
    publish_call = safe[0]["tool_calls"][0]
    assert publish_call["function"]["arguments"] == {"source_path": "/work/report.pdf"}
    assert safe[1]["role"] == "assistant"
    assert safe[2]["role"] == "assistant"
    assert "visible output" in safe[3]["content"]
    assert "tr_42" in safe[3]["content"]


def test_legacy_result_archive_is_sanitized_without_losing_raw_receipt():
    import json
    from agent.core import _sanitize_tool_result_archive_for_model

    raw_value = json.dumps({
        "tool_name": "terminal",
        "arguments": {"command": "pwd"},
        "content": json.dumps({
            "ok": True,
            "result": {
                "_terminal_text": "visible",
                "_terminal_output": "full output",
                "_localagent_internal": {"exit_code": 1, "operation_id": "private-id"},
            },
        }),
    })
    safe, raw = _sanitize_tool_result_archive_for_model({"tr_old": raw_value})
    assert raw["tr_old"] == raw_value
    assert "private-id" not in safe["tr_old"]
    assert "exit_code" not in safe["tr_old"]
    assert "full output" in safe["tr_old"]


def test_stage_reviewer_ignores_successful_plain_terminal_transcript_and_keeps_failure(monkeypatch):
    import json
    from agent import core

    captured = {}
    def fake_chat(messages, tools=None, reasoning=True):
        captured["payload"] = json.loads(messages[1]["content"])
        return {"message": {"content": json.dumps({
            "decision": "continue_step",
            "reason": "One more check is needed.",
            "task_prompt": "Run the next check.",
            "reconciled_steps": [],
        })}}
    monkeypatch.setattr(core, "_client_chat", fake_chat)
    messages = [
        {
            "role": "tool", "tool_name": "terminal", "content": "generated output",
            "_localagent_internal": {
                "state": "completed", "exit_code": 0, "timed_out": False,
                "_terminal_output": "generated output",
            },
        },
        {
            "role": "tool", "tool_name": "terminal", "content": "traceback",
            "_localagent_internal": {
                "state": "failed", "exit_code": 1, "timed_out": False,
                "_terminal_output": "traceback",
            },
        },
    ]
    core._review_step(
        {"goal": "Complete task", "steps": []},
        {"id": "step_1", "expected_result": "Done", "success_criteria": ["Done"]},
        "Need another check.", messages,
    )
    recent = captured["payload"]["recent_interaction"]
    assert len(recent) == 1
    assert recent[0]["role"] == "tool_error"
    assert "traceback" in recent[0]["error"]
    assert "generated output" not in recent[0]["error"]
    assert "exit_code" not in json.dumps(recent, ensure_ascii=False)


def test_tool_result_retrieval_preserves_safe_json_fragments_without_false_redaction():
    from agent.core import _model_visible_tool_result

    fragment = '"operation_id": "ordinary user data",'
    visible = _model_visible_tool_result("get_tool_result", {"content": fragment})
    assert visible["content"] == fragment


def test_audio_support_is_limited_to_gemma4_and_unsupported_payload_is_marked():
    import json
    from agent.core import _mark_loaded_media_not_attached, _model_supports_audio_input

    assert _model_supports_audio_input("gemma4:12b-16k")
    assert _model_supports_audio_input("Gemma4:12b")
    assert not _model_supports_audio_input("qwen3.5:9b")
    messages = [{
        "role": "tool",
        "tool_name": "load_file_for_model",
        "content": json.dumps({
            "ok": True,
            "result": {"kind": "audio", "delivery": "ollama_audio_input_via_wav_images_compatibility"},
            "result_id": "tr_audio",
        }),
    }]
    _mark_loaded_media_not_attached(
        messages, "audio", "Selected model does not support Gemma 4 audio input."
    )
    payload = json.loads(messages[0]["content"])
    assert payload["result_id"] == "tr_audio"
    assert payload["result"]["delivery"] == "not_attached_model_unsupported"
    assert payload["result"]["audio_attached"] is False


def test_unsupported_video_audio_keeps_frame_delivery_metadata():
    import json
    from agent.core import _mark_loaded_media_not_attached

    messages = [{
        "role": "tool",
        "tool_name": "load_file_for_model",
        "content": json.dumps({
            "ok": True,
            "result": {
                "kind": "video",
                "delivery": "sampled_video_frames_as_ollama_images",
                "audio_track_handling": "first_30_seconds_attached_only_when_supported",
                "note": "Video frames sampled.",
            },
            "result_id": "tr_video",
        }),
    }]
    _mark_loaded_media_not_attached(messages, "video", "Gemma 4 audio is unavailable.")
    result = json.loads(messages[0]["content"])["result"]
    assert result["delivery"] == "sampled_video_frames_as_ollama_images"
    assert result["audio_track_handling"] == "not_attached_model_unsupported"
    assert result["audio_attached"] is False
    assert "Video frames sampled." in result["note"]


def test_actual_ollama_overflow_for_text_load_suggests_next_chunk_and_removes_attachment():
    import json
    from agent.core import _replace_overflowed_file_load_with_error

    messages = [
        {
            "role": "tool",
            "tool_name": "load_file_for_model",
            "content": json.dumps({
                "ok": True,
                "result": {"kind": "text", "extracted_text": "large text"},
                "result_id": "tr_text",
            }),
        },
        {
            "role": "user",
            "content": "Loaded file for inspection.",
            "attachments": [{"kind": "text", "path": "/managed/file.txt"}],
        },
    ]
    assert _replace_overflowed_file_load_with_error(messages) is True
    assert len(messages) == 1
    receipt = json.loads(messages[0]["content"])
    assert receipt["result_id"] == "tr_text"
    assert "меньшим length" in receipt["error"]
    assert "start=next_start" in receipt["error"]
    assert receipt["file_not_loaded"] is True


def test_planning_tool_call_summaries_exclude_retired_terminal_ids_and_arguments():
    from agent.core import _model_visible_tool_calls

    calls = [
        {
            "id": "call_old",
            "function": {
                "name": "terminal_poll",
                "arguments": {"operation_id": "internal-op", "cursor": 40},
            },
        },
        {
            "id": "call_new",
            "function": {
                "name": "terminal",
                "arguments": {"command": "pwd", "operation_id": "should-not-leak", "timeout": 10},
            },
        },
        {
            "id": "call_publish",
            "function": {
                "name": "publish_file",
                "arguments": {"source_path": "/workspace/out.txt", "destination_path": "/tmp/out.txt"},
            },
        },
    ]
    visible = _model_visible_tool_calls(calls)
    assert [call["name"] for call in visible] == ["terminal", "publish_file"]
    assert visible[0]["arguments"] == {"command": "pwd", "timeout": 10}
    assert visible[1]["arguments"] == {"source_path": "/workspace/out.txt"}
    assert "operation_id" not in str(visible)
