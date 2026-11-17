from agent.environment import capability_catalog_prompt


def test_model_capability_catalog_is_compact_and_explains_web_search():
    prompt = capability_catalog_prompt()
    assert "terminal" in prompt
    assert 'web-search "query"' in prompt
    assert "Playwright" in prompt
    assert "requests" in prompt
    assert "checked when the container starts" in prompt
    assert "XlsxWriter" not in prompt


def test_shared_manifest_contains_runtime_checks():
    import json
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "container" / "capabilities.json").read_text(encoding="utf-8")
    )
    assert manifest["version"] == 1
    assert any(item["name"] == "Веб-поиск" and "web-search" in item["checks"]["commands"] for item in manifest["capabilities"])
    assert any(item["name"] == "Браузер" and "playwright" in item["checks"]["python_imports"] for item in manifest["capabilities"])
