from types import SimpleNamespace

from openpyxl import Workbook

from agent import tools
from agent.tools import dispatch_tool


def _workspace(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=root))
    return root


def _make_book(path, rows=8):
    wb = Workbook()
    ws = wb.active
    ws.title = "Packers"
    for row in range(1, rows + 1):
        ws.append([f"Part {row}", row, row * 1.5])
    ws2 = wb.create_sheet("Scales")
    ws2.append(["Spare", "Count"])
    ws2.append(["Sensor", 3])
    wb.save(path)
    wb.close()


def test_read_excel_returns_compact_overview_and_sheet_dimensions(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    _make_book(root / "book.xlsx", rows=14)

    result = dispatch_tool("read_excel", {"path": "book.xlsx"})

    assert result["mode"] == "overview"
    assert result["available_sheets"] == ["Packers", "Scales"]
    assert result["sheets"][0]["total_rows"] == 14
    assert result["sheets"][0]["preview"][0] == ["Part 1", 1, 1.5]
    assert result["sheets"][0]["preview_truncated"] is True


def test_read_excel_can_read_next_chunk_by_start_row(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    _make_book(root / "book.xlsx", rows=8)

    first = dispatch_tool("read_excel", {
        "path": "book.xlsx", "sheet_name": "Packers",
        "start_row": 4, "max_rows": 3, "max_cols": 3,
    })["sheets"][0]
    assert first["preview"] == [["Part 4", 4, 6.0], ["Part 5", 5, 7.5], ["Part 6", 6, 9.0]]
    assert first["next_start_row"] == 7
    assert first["has_more_rows"] is True

    second = dispatch_tool("read_excel", {
        "path": "book.xlsx", "sheet_name": "Packers",
        "start_row": first["next_start_row"], "max_rows": 3, "max_cols": 3,
    })["sheets"][0]
    assert second["preview"] == [["Part 7", 7, 10.5], ["Part 8", 8, 12.0]]
    assert second["next_start_row"] is None
    assert second["has_more_rows"] is False


def test_read_excel_caps_cells_per_selected_sheet(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    wb = Workbook()
    ws = wb.active
    ws.title = "Large"
    for row_idx in range(300):
        ws.append([row_idx * 100 + col for col in range(50)])
    wb.save(root / "large.xlsx")
    wb.close()

    sheet = dispatch_tool("read_excel", {
        "path": "large.xlsx", "sheet_name": "Large", "max_rows": 500, "max_cols": 50,
    })["sheets"][0]
    assert sheet["rows_read"] <= 200
    assert sum(len(row) for row in sheet["preview"]) <= 10_000
    assert sheet["has_more_rows"] is True


def test_terminal_is_the_computer_tool_for_excel_and_document_work():
    by_name = {item["function"]["name"]: item["function"] for item in tools.TOOL_DEFINITIONS}
    assert "terminal" in by_name
    description = by_name["terminal"]["description"].lower()
    assert "python" in description and "linux" in description
    assert "read_excel" not in by_name and "write_excel" not in by_name

    from agent.core import SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT
    assert "terminal" in SYSTEM_PROMPT
    assert "Do not manually parse XLSX ZIP/XML" in PLAN_SYSTEM_PROMPT


def test_office_input_size_limit_is_50_mb(tmp_path, monkeypatch):
    root = _workspace(tmp_path, monkeypatch)
    path = root / "oversized.xlsx"
    path.write_bytes(b"x" * 32)
    monkeypatch.setattr(tools, "MAX_OFFICE_FILE_BYTES", 16)

    import pytest
    with pytest.raises(ValueError, match="50 MB"):
        dispatch_tool("read_excel", {"path": "oversized.xlsx"})
