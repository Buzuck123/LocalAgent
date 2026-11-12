import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import tools


def setup_workspace(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service_db = tmp_path / "service" / "agent.sqlite3"
    service_db.parent.mkdir()
    monkeypatch.setattr(tools, "settings", SimpleNamespace(workspace_path=workspace, database_path=service_db))
    return workspace, service_db


def test_sqlite_create_mutate_filter_aggregate_and_parameterization(monkeypatch, tmp_path):
    workspace, _ = setup_workspace(monkeypatch, tmp_path)
    assert tools.sqlite_create_database("sales.db")["created"] is True
    tools.sqlite_execute("sales.db", "CREATE TABLE sales (id INTEGER PRIMARY KEY, item TEXT, amount INTEGER)")
    tools.sqlite_execute("sales.db", "INSERT INTO sales(item, amount) VALUES (?, ?)", ["O'Reilly", 12])
    tools.sqlite_execute("sales.db", "INSERT INTO sales(item, amount) VALUES (?, ?)", ["Paper", 20])

    result = tools.sqlite_query(
        "sales.db", action="select", table_name="sales",
        filters=[{"column": "item", "operator": "=", "value": "O'Reilly"}],
    )
    assert result["returned_rows"] == 1
    assert result["rows"][0]["amount"] == 12
    paged = tools.sqlite_query("sales.db", action="sql", sql="SELECT id, item FROM sales ORDER BY id", limit=1, offset=1)
    assert paged["rows"] == [{"id": 2, "item": "Paper"}]

    malicious = tools.sqlite_query(
        "sales.db", action="select", table_name="sales",
        filters=[{"column": "item", "operator": "=", "value": "x' OR 1=1 --"}],
    )
    assert malicious["returned_rows"] == 0

    aggregate = tools.sqlite_query(
        "sales.db", action="aggregate", table_name="sales", group_by=[],
        aggregations=[{"function": "SUM", "column": "amount", "alias": "total"}],
    )
    assert aggregate["rows"] == [{"total": 32}]
    assert Path(workspace, "sales.db").is_file()


def test_service_database_is_read_only_and_attach_is_blocked(monkeypatch, tmp_path):
    workspace, service_db = setup_workspace(monkeypatch, tmp_path)
    with sqlite3.connect(service_db) as con:
        con.execute("CREATE TABLE service_info (value TEXT)")
        con.execute("INSERT INTO service_info VALUES ('safe')")

    result = tools.sqlite_query("agent", action="select", table_name="service_info")
    assert result["rows"] == [{"value": "safe"}]
    with pytest.raises(ValueError):
        tools.sqlite_execute("agent", "DELETE FROM service_info")

    tools.sqlite_create_database("user.db")
    tools.sqlite_execute("user.db", "CREATE TABLE t (value TEXT)")
    with pytest.raises(ValueError, match="not authorized|rejected"):
        tools.sqlite_execute("user.db", f"ATTACH DATABASE '{service_db}' AS other")
    with pytest.raises(ValueError, match="rejected|failed"):
        tools.sqlite_query("user.db", action="sql", sql="DELETE FROM t")
    assert tools.sqlite_query("agent", action="select", table_name="service_info")["rows"] == [{"value": "safe"}]


def test_database_paths_cannot_escape_workspace_or_overwrite(monkeypatch, tmp_path):
    workspace, _ = setup_workspace(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        tools.sqlite_create_database("../outside.db")
    tools.sqlite_create_database("safe.db")
    with pytest.raises(FileExistsError):
        tools.sqlite_create_database("safe.db")


def test_query_result_is_bounded_by_characters(monkeypatch, tmp_path):
    setup_workspace(monkeypatch, tmp_path)
    tools.sqlite_create_database("large.db")
    tools.sqlite_execute("large.db", "CREATE TABLE items (id INTEGER, payload TEXT)")
    for i in range(50):
        tools.sqlite_execute("large.db", "INSERT INTO items VALUES (?, ?)", [i, "x" * 500])
    result = tools.sqlite_query("large.db", action="select", table_name="items", max_chars=1000, limit=1000)
    assert result["truncated"] is True
    assert result["returned_rows"] < 50
    assert result["returned_rows"] > 0


def test_excel_roundtrip_through_sqlite(monkeypatch, tmp_path):
    workspace, _ = setup_workspace(monkeypatch, tmp_path)
    from openpyxl import Workbook, load_workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Products"
    sheet.append(["SKU", "Price", "Name"])
    sheet.append(["A-1", 10.5, "=1+1"])
    sheet["C2"].data_type = "s"
    sheet.append(["B-2", 20, "Notebook"])
    workbook.save(workspace / "products.xlsx")
    workbook.close()

    tools.sqlite_create_database("products.db")
    imported = tools.sqlite_import_excel("products.db", "products.xlsx", "products", "Products")
    assert imported["rows_imported"] == 2
    assert imported["columns"] == ["SKU", "Price", "Name"]

    result = tools.sqlite_query("products.db", action="aggregate", table_name="products",
                                aggregations=[{"function": "SUM", "column": "Price", "alias": "total"}])
    assert result["rows"] == [{"total": 30.5}]

    exported = tools.sqlite_export_excel("products.db", 'SELECT SKU, Price, Name FROM products ORDER BY SKU', "export.xlsx")
    assert exported["rows_exported"] == 2
    out = load_workbook(workspace / "export.xlsx", data_only=False)
    try:
        sheet = out["Data"]
        assert sheet["C2"].value == "=1+1"
        assert sheet["C2"].data_type == "s"
        assert sheet.max_row == 3
    finally:
        out.close()


def test_sqlite_operations_are_available_through_terminal_not_specialized_tools():
    names = {item["function"]["name"] for item in tools.TOOL_DEFINITIONS}
    assert "terminal" in names
    assert not {"sqlite_create_database", "sqlite_query", "sqlite_execute", "sqlite_import_excel", "sqlite_export_excel"} & names
