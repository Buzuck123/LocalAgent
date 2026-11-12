from agent.core import _diagnose_tool_failure


def test_existing_file_error_guides_agent_to_inspect_instead_of_repeat():
    state = {}
    result = _diagnose_tool_failure("terminal", {"command": "touch a.txt"}, "FileExistsError: a.txt", state)
    assert result["category"] == "existing_resource"
    assert "read its contents" in result["guidance"]
    assert result["repeat_count"] == 1


def test_identical_failure_is_marked_as_repeat():
    state = {}
    args = {"path": "/workspace/a.txt"}
    _diagnose_tool_failure("load_file_for_model", args, "FileNotFoundError: a.txt", state)
    result = _diagnose_tool_failure("load_file_for_model", args, "FileNotFoundError: a.txt", state)
    assert result["same_failure_repeated"] is True
    assert result["repeat_count"] == 2
    assert "This repeats" in result["guidance"]


def test_import_failure_recommends_installing_or_replacing_module():
    result = _diagnose_tool_failure("terminal", {"command": "python script.py"}, "ModuleNotFoundError: No module named 'x'", {})
    assert result["category"] == "import_or_environment"
    assert "Install it if needed" in result["guidance"]


def test_repeated_failure_warning_recommends_changing_strategy():
    state = {}
    args = {"command": "python script.py"}
    _diagnose_tool_failure("terminal", args, "ValueError: old path error", state)
    _diagnose_tool_failure("terminal", args, "ValueError: old path error", state)
    result = _diagnose_tool_failure("terminal", args, "ValueError: old path error", state)
    assert result["repeat_count"] == 3
    assert "choose a different tool" in result["guidance"]
