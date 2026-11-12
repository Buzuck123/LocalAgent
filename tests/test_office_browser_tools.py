import pytest
from types import SimpleNamespace
from openpyxl import load_workbook
from docx import Document

from agent import tools


def _settings(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    workspace.mkdir(exist_ok=True)
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=workspace))
    return workspace


def test_excel_write_read_and_literal_formula(tmp_path, monkeypatch):
    workspace = _settings(tmp_path, monkeypatch)
    created = tools.write_excel(
        "report.xlsx", "Summary",
        [["Name", "Value"], ["=HYPERLINK(\"https://example.invalid\")", 42]],
    )
    assert created["rows_written"] == 2
    result = tools.read_excel("report.xlsx", "Summary")
    assert result["sheets"][0]["preview"][0] == ["Name", "Value"]
    assert result["sheets"][0]["preview"][1] == ['=HYPERLINK("https://example.invalid")', 42]
    wb = load_workbook(workspace / "report.xlsx", data_only=False)
    try:
        assert wb["Summary"]["A2"].data_type == "s"
    finally:
        wb.close()


def test_excel_existing_cells_need_overwrite(tmp_path, monkeypatch):
    _settings(tmp_path, monkeypatch)
    tools.write_excel("report.xlsx", "Sheet1", [[1]])
    with pytest.raises(FileExistsError):
        tools.write_excel("report.xlsx", "Sheet1", [[2]])
    tools.write_excel("report.xlsx", "Sheet1", [[2]], overwrite=True)
    result = tools.read_excel("report.xlsx", "Sheet1")
    assert result["sheets"][0]["preview"][0][0] == 2


def test_office_helpers_reject_path_escape_and_wrong_extension(tmp_path, monkeypatch):
    _settings(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        tools.read_excel("../outside.xlsx")
    with pytest.raises(ValueError):
        tools.write_word("bad.txt", ["x"])


def test_word_round_trip_with_table(tmp_path, monkeypatch):
    workspace = _settings(tmp_path, monkeypatch)
    result = tools.write_word(
        "summary.docx", ["Report", "Second paragraph"],
        [[["Name", "Count"], ["Apples", 3]]],
    )
    assert result["tables_written"] == 1
    read = tools.read_word("summary.docx")
    assert read["paragraphs"] == ["Report", "Second paragraph"]
    assert read["tables"][0][1] == ["Apples", "3"]
    doc = Document(workspace / "summary.docx")
    assert len(doc.tables) == 1


def test_word_existing_file_is_protected(tmp_path, monkeypatch):
    _settings(tmp_path, monkeypatch)
    tools.write_word("summary.docx", ["first"])
    with pytest.raises(FileExistsError):
        tools.write_word("summary.docx", ["second"])


def test_public_url_validator_blocks_local_and_nonstandard_ports(monkeypatch):
    monkeypatch.setattr(tools.socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("93.184.216.34", 0))])
    with pytest.raises(ValueError):
        tools._validate_public_url("http://127.0.0.1/")
    with pytest.raises(ValueError):
        tools._validate_public_url("https://example.com:8443/")
    assert tools._validate_public_url("https://example.com/") == "https://example.com/"


def test_browser_tool_remains_and_office_helpers_are_not_model_tools():
    names = {item["function"]["name"] for item in tools.TOOL_DEFINITIONS}
    assert {"terminal", "publish_file", "open_browser_page"} <= names
    assert not {"read_excel", "write_excel", "read_word", "write_word"} & names
