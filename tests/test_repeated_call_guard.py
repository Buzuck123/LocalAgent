from agent.orchestration import should_block_repeated_call, tool_call_fingerprint


def test_fingerprint_ignores_dict_key_order():
    assert tool_call_fingerprint("terminal", {"command": "pwd", "timeout": 20}) == tool_call_fingerprint(
        "terminal", {"timeout": 20, "command": "pwd"}
    )


def test_first_call_is_not_blocked():
    assert should_block_repeated_call("load_file_for_model", {"path": "/workspace/book.pdf"}, []) is None


def test_exact_call_after_failure_is_blocked():
    args = {"command": "cat /workspace/missing.txt"}
    history = [{
        "tool_name": "terminal",
        "call_fingerprint": tool_call_fingerprint("terminal", args),
        "error": "No such file",
    }]
    result = should_block_repeated_call("terminal", args, history)
    assert result["blocked"] is True
    assert result["category"] == "repeated_failed_call"
    assert result["retry_safe_without_state_check"] is False


def test_changed_arguments_are_not_blocked():
    history = [{
        "call_fingerprint": tool_call_fingerprint("terminal", {"command": "pwd"}),
        "error": "bad command",
    }]
    assert should_block_repeated_call("terminal", {"command": "ls"}, history) is None


def test_non_list_history_is_safe():
    assert should_block_repeated_call("load_file_for_model", {"path": "/workspace/x.txt"}, None) is None


def test_unrelated_terminal_success_does_not_unblock_same_failed_call():
    args = {"command": "cat /workspace/missing.txt"}
    other = {"command": "ls /workspace"}
    history = [{
        "tool_name": "terminal",
        "call_fingerprint": tool_call_fingerprint("terminal", args),
        "error": "No such file",
    }, {
        "status": "success",
        "tool_name": "terminal",
        "call_fingerprint": tool_call_fingerprint("terminal", other),
        "arguments": other,
    }]
    assert should_block_repeated_call("terminal", args, history)["blocked"] is True


def test_same_call_success_after_failure_clears_guard():
    args = {"command": "pwd"}
    fingerprint = tool_call_fingerprint("terminal", args)
    history = [{
        "tool_name": "terminal",
        "call_fingerprint": fingerprint,
        "error": "temporary error",
    }, {
        "status": "success",
        "tool_name": "terminal",
        "call_fingerprint": fingerprint,
    }]
    assert should_block_repeated_call("terminal", args, history) is None


def test_terminal_creation_of_missing_file_unblocks_model_inspection_retry():
    failed_args = {"path": "/work/read_verify_files.txt"}
    repair_args = {"command": "touch /work/read_verify_files.txt"}
    history = [{
        "tool_name": "load_file_for_model",
        "arguments": failed_args,
        "call_fingerprint": tool_call_fingerprint("load_file_for_model", failed_args),
        "error": "FileNotFoundError: /work/read_verify_files.txt",
    }, {
        "status": "success",
        "tool_name": "terminal",
        "arguments": repair_args,
        "call_fingerprint": tool_call_fingerprint("terminal", repair_args),
    }]
    assert should_block_repeated_call("load_file_for_model", failed_args, history) is None


def test_unrelated_file_creation_does_not_unblock_model_inspection_retry():
    failed_args = {"path": "/work/read_verify_files.txt"}
    repair_args = {"command": "touch /work/other.txt"}
    history = [{
        "tool_name": "load_file_for_model",
        "arguments": failed_args,
        "call_fingerprint": tool_call_fingerprint("load_file_for_model", failed_args),
        "error": "FileNotFoundError: /work/read_verify_files.txt",
    }, {
        "status": "success",
        "tool_name": "terminal",
        "arguments": repair_args,
        "call_fingerprint": tool_call_fingerprint("terminal", repair_args),
    }]
    assert should_block_repeated_call("load_file_for_model", failed_args, history)["blocked"] is True


def test_excel_strategy_remains_in_planner_and_step_prompts():
    from agent.core import PLAN_SYSTEM_PROMPT, SYSTEM_PROMPT
    assert "inspect workbook structure first" in PLAN_SYSTEM_PROMPT
    assert "process only relevant sheets" in PLAN_SYSTEM_PROMPT
    assert "do not load an entire workbook into model context" in PLAN_SYSTEM_PROMPT
    assert "terminal" in SYSTEM_PROMPT
    assert "publish_file" in SYSTEM_PROMPT


def test_resume_refreshes_old_agent_system_prompt_only():
    from agent.core import SYSTEM_PROMPT, _refresh_saved_system_prompt
    messages = [
        {"role": "system", "content": "Ты — локальный ИИ-агент пользователя. Старые правила."},
        {"role": "system", "content": "Контекстная сводка: оставить как есть"},
        {"role": "user", "content": "Продолжи задачу"},
    ]
    result = _refresh_saved_system_prompt(messages)
    assert result[0]["content"] == SYSTEM_PROMPT
    assert result[1]["content"] == "Контекстная сводка: оставить как есть"
    assert result[2]["content"] == "Продолжи задачу"


def test_work_prefix_without_leading_slash_is_normalized_for_file_paths():
    from agent.tools import _work_path_suffix
    assert _work_path_suffix("work/create.py") == "create.py"
    assert _work_path_suffix("/work/create.py") == "create.py"
    assert _work_path_suffix("create.py") is None
