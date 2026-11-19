from agent import core
from agent.llama_cpp import LlamaCppClient


def test_ollama_profile_uses_loaded_runtime_context_not_architecture_limit(monkeypatch):
    client = core.OllamaClient.__new__(core.OllamaClient)
    client._post = lambda *args, **kwargs: {
        "parameters": "num_ctx 32768\nnum_predict 4096",
        "model_info": {"qwen3.context_length": 262144},
    }
    client.status = lambda: {
        "models": [{"name": "qwen3.5:9b-32k", "context_length": 65536}]
    }

    profile = client.model_profile("qwen3.5:9b-32k")

    assert profile["num_ctx"] == 65536
    assert profile["runtime_context_limit"] == 65536
    assert profile["model_context_length"] == 262144
    assert profile["num_predict"] == 4096
    assert profile["runtime_context_confirmed"] is True
    assert profile["source"] == "ollama_runtime"


def test_ollama_profile_uses_modelfile_context_before_model_is_loaded():
    client = core.OllamaClient.__new__(core.OllamaClient)
    client._post = lambda *args, **kwargs: {
        "parameters": "num_ctx 32768",
        "model_info": {"qwen3.context_length": 262144},
    }
    client.status = lambda: {"models": []}

    profile = client.model_profile("qwen3.5:9b")

    assert profile["num_ctx"] == 32768
    assert profile["model_context_length"] == 262144
    assert profile["runtime_context_confirmed"] is False
    assert profile["source"] == "modelfile"


def test_ollama_profile_marks_localagent_fallback_as_unconfirmed():
    client = core.OllamaClient.__new__(core.OllamaClient)
    client._post = lambda *args, **kwargs: {
        "parameters": "",
        "model_info": {"qwen3.context_length": 262144},
    }
    client.status = lambda: {"models": []}

    profile = client.model_profile("qwen3.5:9b")

    assert profile["num_ctx"] == core.settings.ollama_context_fallback
    assert profile["model_context_length"] == 262144
    assert profile["runtime_context_confirmed"] is False
    assert profile["source"] == "localagent_fallback_unconfirmed"


def test_llama_cpp_profile_uses_server_props_not_training_context():
    client = LlamaCppClient.__new__(LlamaCppClient)
    client.model = "bonsai"
    client._available_models = lambda: [{
        "name": "bonsai",
        "context_length": 262144,
        "model_context_length": 262144,
    }]
    client._server_json_request = lambda path: {
        "default_generation_settings": {"n_ctx": 65536}
    }

    profile = client.model_profile("bonsai")

    assert profile["num_ctx"] == 65536
    assert profile["runtime_context_limit"] == 65536
    assert profile["model_context_length"] == 262144
    assert profile["runtime_context_confirmed"] is True
    assert profile["source"] == "llama_cpp_runtime_props"



def test_llama_cpp_profile_uses_router_runtime_context_when_props_is_not_proxied():
    client = LlamaCppClient.__new__(LlamaCppClient)
    client.model = "bonsai"
    client._available_models = lambda: [{
        "name": "bonsai",
        "runtime_context_limit": 65536,
        "model_context_length": 262144,
    }]
    client._server_json_request = lambda path: {}

    profile = client.model_profile("bonsai")

    assert profile["num_ctx"] == 65536
    assert profile["model_context_length"] == 262144
    assert profile["runtime_context_confirmed"] is True
    assert profile["source"] == "llama_cpp_models_runtime"

def test_llama_cpp_profile_fallback_does_not_use_training_context():
    client = LlamaCppClient.__new__(LlamaCppClient)
    client.model = "bonsai"
    client._available_models = lambda: [{
        "name": "bonsai",
        "context_length": 262144,
        "model_context_length": 262144,
    }]
    def unavailable(_path):
        raise RuntimeError("props unavailable")
    client._server_json_request = unavailable

    profile = client.model_profile("bonsai")

    assert profile["num_ctx"] == core.settings.llama_cpp_context
    assert profile["model_context_length"] == 262144
    assert profile["runtime_context_confirmed"] is False
    assert profile["source"] == "llama_cpp_config_fallback_unconfirmed"


def test_llama_cpp_server_native_url_is_outside_openai_v1_prefix():
    client = LlamaCppClient.__new__(LlamaCppClient)
    client.base = "http://127.0.0.1:8081/v1"

    assert client._server_url("/props") == "http://127.0.0.1:8081/props"
