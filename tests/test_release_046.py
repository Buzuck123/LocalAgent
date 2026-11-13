from types import SimpleNamespace

from agent import core, tools


def _names(definitions):
    return {item["function"]["name"] for item in definitions}


def test_bounded_python_script_task_skips_planner():
    goal = (
        "Создай Python-скрипт в /work, который создаст Excel-файл с двумя листами, "
        "заполнит данными, сохранит общую таблицу в CSV и проверит результат. "
        "Если что-то не получится, попробуй исправить."
    )
    assert core._is_bounded_python_script_workflow(goal) is True
    assert core.classify_execution_mode(goal) == "simple"


def test_python_script_task_keeps_full_registry_available():
    goal = (
        "Создай Python-скрипт в /work, который создаст Excel-файл и CSV, "
        "запусти и проверь результат."
    )
    names = _names(core.tools_for_task(goal))
    assert {"terminal", "publish_file"} <= names
    assert not {"web_search", "open_browser_page"} & names
    assert not {"write_file", "run_python", "read_file", "list_files", "read_output_file",
                "read_excel", "write_excel", "excel_aggregate", "sqlite_query"} & names


def test_complex_python_project_still_uses_planner():
    goal = "Разработай полноценное Python-приложение с веб-интерфейсом и базой данных"
    assert core._is_bounded_python_script_workflow(goal) is False
    assert core.classify_execution_mode(goal) == "planned"
