from agent.orchestration import should_block_repeated_call, tool_call_fingerprint


def test_fingerprint_ignores_dict_key_order():
    assert tool_call_fingerprint("sqlite_query", {"a": 1, "b": 2}) == tool_call_fingerprint(
        "sqlite_query", {"b": 2, "a": 1}
    )


def test_first_call_is_not_blocked():
    assert should_block_repeated_call("read_excel", {"path": "book.xlsx"}, []) is None


def test_exact_call_after_failure_is_blocked():
    args = {"database_path": "dbs/data.db", "sql": "SELECT strval(x) FROM t"}
    history = [{
        "tool_name": "sqlite_query",
        "call_fingerprint": tool_call_fingerprint("sqlite_query", args),
        "error": "SQLite query failed: no such function: strval",
    }]
    result = should_block_repeated_call("sqlite_query", args, history)
    assert result["blocked"] is True
    assert result["category"] == "repeated_failed_call"
    assert result["retry_safe_without_state_check"] is False
    assert "no such function" in result["previous_error"]


def test_changed_arguments_are_not_blocked():
    history = [{
        "call_fingerprint": tool_call_fingerprint("sqlite_query", {"sql": "SELECT 1"}),
        "error": "bad query",
    }]
    assert should_block_repeated_call("sqlite_query", {"sql": "SELECT 2"}, history) is None


def test_non_list_history_is_safe():
    assert should_block_repeated_call("read_excel", {"path": "x.xlsx"}, None) is None


def test_unrelated_success_does_not_unblock_same_failed_call():
    args = {"path": "missing.xlsx"}
    history = [{
        "tool_name": "read_excel",
        "call_fingerprint": tool_call_fingerprint("read_excel", args),
        "error": "file not found",
    }, {
        "status": "success",
        "tool_name": "list_files",
        "call_fingerprint": tool_call_fingerprint("list_files", {"path": "."}),
    }]
    result = should_block_repeated_call("read_excel", args, history)
    assert result["blocked"] is True


def test_same_call_success_after_failure_clears_guard():
    args = {"path": "missing.xlsx"}
    fingerprint = tool_call_fingerprint("read_excel", args)
    history = [{
        "tool_name": "read_excel",
        "call_fingerprint": fingerprint,
        "error": "file not found",
    }, {
        "status": "success",
        "tool_name": "read_excel",
        "call_fingerprint": fingerprint,
    }]
    assert should_block_repeated_call("read_excel", args, history) is None


def test_excel_strategy_is_present_in_planner_and_step_prompts():
    from agent.core import PLAN_SYSTEM_PROMPT, STEP_TASK_PROMPT_SYSTEM
    assert "inspect workbook structure first" in PLAN_SYSTEM_PROMPT
    assert "process only relevant sheets" in PLAN_SYSTEM_PROMPT
    assert "do not load an entire workbook into context" in STEP_TASK_PROMPT_SYSTEM
    assert "publish_file" in STEP_TASK_PROMPT_SYSTEM or "terminal" in STEP_TASK_PROMPT_SYSTEM


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


def test_matching_file_creation_unblocks_previously_missing_python_script():
    failed_args = {"path": "/work/read_verify_files.py", "timeout_seconds": 60}
    write_args = {"path": "/work/read_verify_files.py", "content": "print('ok')"}
    history = [{
        "tool_name": "run_python",
        "arguments": failed_args,
        "call_fingerprint": tool_call_fingerprint("run_python", failed_args),
        "error": "FileNotFoundError: Python script not found: /work/read_verify_files.py",
    }, {
        "status": "success",
        "tool_name": "write_file",
        "arguments": write_args,
        "call_fingerprint": tool_call_fingerprint("write_file", write_args),
    }]
    assert should_block_repeated_call("run_python", failed_args, history) is None


def test_unrelated_file_creation_does_not_unblock_missing_script_retry():
    failed_args = {"path": "/work/read_verify_files.py", "timeout_seconds": 60}
    write_args = {"path": "/work/other.py", "content": "print('other')"}
    history = [{
        "tool_name": "run_python",
        "arguments": failed_args,
        "call_fingerprint": tool_call_fingerprint("run_python", failed_args),
        "error": "FileNotFoundError: Python script not found: /work/read_verify_files.py",
    }, {
        "status": "success",
        "tool_name": "write_file",
        "arguments": write_args,
        "call_fingerprint": tool_call_fingerprint("write_file", write_args),
    }]
    assert should_block_repeated_call("run_python", failed_args, history)["blocked"] is True



def test_work_prefix_without_leading_slash_is_normalized_for_file_paths():
    from agent.tools import _work_path_suffix
    assert _work_path_suffix("work/create.py") == "create.py"
    assert _work_path_suffix("/work/create.py") == "create.py"
    assert _work_path_suffix("create.py") is None


def test_rewriting_script_after_python_exception_unblocks_same_run_call():
    failed_args = {"path": "create_excel_and_csv.py"}
    write_args = {"path": "create_excel_and_csv.py", "content": "fixed code", "overwrite": True}
    history = [{
        "tool_name": "run_python",
        "arguments": failed_args,
        "call_fingerprint": tool_call_fingerprint("run_python", failed_args),
        "error": "AttributeError: property 'active' of 'Workbook' object has no deleter",
    }, {
        "status": "success",
        "tool_name": "write_file",
        "arguments": write_args,
        "call_fingerprint": tool_call_fingerprint("write_file", write_args),
    }]
    assert should_block_repeated_call("run_python", failed_args, history) is None


def test_work_shorthand_matches_explicit_work_path_for_recovery():
    failed_args = {"path": "/work/create.py"}
    write_args = {"path": "work/create.py", "content": "fixed"}
    history = [{
        "tool_name": "run_python",
        "arguments": failed_args,
        "call_fingerprint": tool_call_fingerprint("run_python", failed_args),
        "error": "SyntaxError: invalid syntax",
    }, {
        "status": "success",
        "tool_name": "write_file",
        "arguments": write_args,
        "call_fingerprint": tool_call_fingerprint("write_file", write_args),
    }]
    assert should_block_repeated_call("run_python", failed_args, history) is None
