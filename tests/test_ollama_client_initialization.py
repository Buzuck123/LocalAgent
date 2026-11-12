from urllib.request import OpenerDirector

from agent.core import OllamaClient


def test_ollama_client_initializes_direct_http_opener():
    client = OllamaClient()
    assert isinstance(client.opener, OpenerDirector)
    assert client.base


def test_cancel_generation_without_active_response_is_safe():
    client = OllamaClient()
    assert client.cancel_generation("missing-task") is False
