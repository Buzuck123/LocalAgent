import json

from agent import core


def tool_message(name, result, ok=True):
    return {
        "role": "tool",
        "tool_name": name,
        "content": json.dumps({"ok": ok, "result": result}, ensure_ascii=False),
    }


def test_action_goal_cannot_complete_without_successful_tool_evidence():
    goal = "Создай файл report.txt"
    messages = [{"role": "user", "content": goal}]
    verdict = {
        "decision": "complete",
        "requires_action": True,
        "evidence_tools": [],
        "reason": "Ассистент сообщил, что файл создан.",
    }

    accepted, reason = core._validate_autonomous_completion(goal, messages, verdict)

    assert accepted is False
    assert "не указал" in reason


def test_failed_tool_result_cannot_be_used_as_completion_evidence():
    goal = "Создай файл report.txt"
    messages = [tool_message("write_file", {"error": "permission denied"}, ok=False)]
    verdict = {
        "decision": "complete",
        "requires_action": True,
        "evidence_tools": ["write_file"],
        "reason": "Файл создан.",
    }

    accepted, _ = core._validate_autonomous_completion(goal, messages, verdict)

    assert accepted is False


def test_successful_tool_evidence_allows_action_completion():
    goal = "Посчитай 2+2"
    messages = [tool_message("calculate", {"result": 4})]
    verdict = {
        "decision": "complete",
        "requires_action": True,
        "evidence_tools": ["calculate"],
        "reason": "Результат подтверждён калькулятором.",
    }

    accepted, reason = core._validate_autonomous_completion(goal, messages, verdict)

    assert accepted is True
    assert "подтверждён" in reason


def test_information_answer_can_complete_without_tool_calls():
    goal = "Что такое рекурсия?"
    verdict = {
        "decision": "complete",
        "requires_action": False,
        "evidence_tools": [],
        "reason": "Дан прямой информационный ответ.",
    }

    accepted, _ = core._validate_autonomous_completion(
        goal, [{"role": "assistant", "content": "Рекурсия — это..."}], verdict
    )

    assert accepted is True


def test_unified_mode_is_persisted_at_submission(tmp_path, monkeypatch):
    from dataclasses import replace
    from agent import db

    db_settings = replace(db.settings, database_path=tmp_path / "agent.sqlite3")
    agent_settings = replace(
        core.settings,
        database_path=tmp_path / "agent.sqlite3",
        workspace_path=tmp_path / "workspace",
    )
    agent_settings.workspace_path.mkdir()
    monkeypatch.setattr(db, "settings", db_settings)
    monkeypatch.setattr(core, "settings", agent_settings)
    db.init_db()
    conversation = db.create_conversation("mode test")

    # Avoid starting a worker; test only the durable checkpoint written on submit.
    agent = core.Agent()
    monkeypatch.setattr(agent, "_run_task", lambda *args: None)
    task = agent.submit(conversation["id"], "Что такое рекурсия?")
    import time
    thread = agent.threads[task["id"]]
    thread.join(timeout=1)

    checkpoint = json.loads(db.get_task(task["id"])["checkpoint_json"])
    assert checkpoint["mode"] == "unified"


def test_filesystem_mutation_requires_later_observation():
    goal = "Создай файл report.txt"
    messages = [tool_message("write_file", {"path": "report.txt"})]
    verdict = {
        "decision": "complete",
        "requires_action": True,
        "evidence_tools": ["write_file"],
        "reason": "Запись прошла успешно.",
    }

    accepted, reason = core._validate_autonomous_completion(goal, messages, verdict)

    assert accepted is False
    assert "последующим чтением" in reason


def test_filesystem_mutation_with_later_read_can_be_verified():
    goal = "Создай файл report.txt"
    messages = [
        tool_message("write_file", {"path": "report.txt"}),
        tool_message("read_file", {"path": "report.txt", "content": "hello"}),
    ]
    verdict = {
        "decision": "complete",
        "requires_action": True,
        "evidence_tools": ["write_file", "read_file"],
        "reason": "Файл создан и прочитан после записи.",
    }

    accepted, _ = core._validate_autonomous_completion(goal, messages, verdict)

    assert accepted is True


def test_same_action_detection_counts_repeated_arguments_even_if_result_varies():
    action_key = '{"name": "read_file", "arguments": {"path": "data.txt"}}'
    history = [
        action_key + ' => {"ok": true, "result": "A"}',
        '{"name": "list_files", "arguments": {}} => {"ok": true}',
        action_key + ' => {"ok": true, "result": "B"}',
    ]

    assert core._same_action_count(action_key, history) == 2


def test_same_arguments_with_changing_results_do_not_trigger_hard_loop_pause():
    # A script/test command can legitimately be called again after the files it
    # reads have changed. Repeated arguments are a warning signal only.
    assert core._should_pause_for_repetition(repeat_count=1, pause_threshold=6) is False


def test_identical_consecutive_action_and_result_triggers_hard_loop_pause():
    assert core._should_pause_for_repetition(repeat_count=6, pause_threshold=6) is True
