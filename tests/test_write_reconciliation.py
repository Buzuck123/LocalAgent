import pytest
from types import SimpleNamespace

from agent import tools


def test_write_file_creates_new_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tools,
        "settings",
        SimpleNamespace(workspace_path=tmp_path.resolve()),
    )

    result = tools.write_workspace_file("result.txt", "expected", overwrite=False)

    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "expected"
    assert result["path"] == "result.txt"
    assert result["overwritten"] is False


def test_write_file_does_not_overwrite_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tools,
        "settings",
        SimpleNamespace(workspace_path=tmp_path.resolve()),
    )
    target = tmp_path / "result.txt"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError):
        tools.write_workspace_file("result.txt", "expected", overwrite=False)

    assert target.read_text(encoding="utf-8") == "original"


def test_matching_file_content_alone_does_not_prove_operation_completed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        tools,
        "settings",
        SimpleNamespace(workspace_path=tmp_path.resolve()),
    )
    target = tmp_path / "result.txt"

    # Файл мог быть создан другим процессом до восстановления задачи.
    target.write_text("expected", encoding="utf-8")

    assert target.exists()
    assert target.read_text(encoding="utf-8") == "expected"
    # Намеренно нет утверждения, что прерванная операция выполнилась.
