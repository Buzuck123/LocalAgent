"""Shared pytest fixtures for deterministic provider selection."""
import pytest

from agent import core


@pytest.fixture(autouse=True)
def use_ollama_as_default_test_provider(monkeypatch):
    """Prevent the developer's persisted provider setting from leaking into tests.

    Tests that exercise llama.cpp or provider switching can explicitly replace
    these attributes; ordinary tests and FakeClient-based tests get a matching
    Ollama provider/client pair by default.
    """
    monkeypatch.setattr(core, "active_llm_provider", "ollama")
    monkeypatch.setattr(core, "client", core.OllamaClient())
