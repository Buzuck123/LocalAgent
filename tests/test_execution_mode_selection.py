import json

from agent import core


class ModeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, messages, tools=None, reasoning=True):
        self.calls.append((messages, tools, reasoning))
        return {"message": {"content": self.response}}


def test_llm_can_select_direct_mode(monkeypatch):
    fake = ModeClient(json.dumps({"mode": "direct", "reason": "Simple inspection"}))
    monkeypatch.setattr(core, "client", fake)
    assert core.select_execution_mode_with_llm("Что написано в документе?") == "simple"
    assert len(fake.calls) == 1
    assert fake.calls[0][1] is None


def test_llm_can_select_plan_mode_for_multistage_work(monkeypatch):
    fake = ModeClient(json.dumps({"mode": "plan", "reason": "The reports depend on the research stage"}))
    monkeypatch.setattr(core, "client", fake)
    assert core.select_execution_mode_with_llm("Сначала исследуй источники, затем подготовь на их основе три взаимосвязанных отчёта") == "planned"


def test_invalid_llm_mode_uses_safe_classifier_fallback(monkeypatch):
    fake = ModeClient("not json")
    monkeypatch.setattr(core, "client", fake)
    assert core.select_execution_mode_with_llm("Что такое фотосинтез") == "simple"
    assert core.select_execution_mode_with_llm("Обработай данные, сгруппируй их и подготовь отчёт") == "planned"


def test_llm_can_choose_direct_for_multiple_independent_files(monkeypatch):
    fake = ModeClient(json.dumps({"mode": "direct", "reason": "Two independent outputs"}))
    monkeypatch.setattr(core, "client", fake)
    assert core.select_execution_mode_with_llm("Создай два независимых текстовых файла") == "simple"



def test_execution_mode_selection_logs_llm_reason(monkeypatch):
    fake = ModeClient(json.dumps({"mode": "plan", "reason": "Research requires several independent checks"}))
    monkeypatch.setattr(core, "client", fake)
    recorded = []
    monkeypatch.setattr(core.db, "record_debug_trace", lambda task_id, event_type, payload: recorded.append((event_type, payload)))
    assert core.select_execution_mode_with_llm("Сравни несколько вариантов") == "planned"
    assert recorded[-1][0] == "execution_mode_selected"
    assert recorded[-1][1]["reason"] == "Research requires several independent checks"
    assert recorded[-1][1]["source"] == "llm"


def test_execution_mode_prompt_distinguishes_task_complexity_from_tool_count():
    prompt = core.EXECUTION_MODE_PROMPT
    assert "several independent checks" in prompt
    assert "comparing alternatives" in prompt
    assert "multiple sources" in prompt
    assert "number of tool calls" in prompt
    assert "merely for formality" in prompt
