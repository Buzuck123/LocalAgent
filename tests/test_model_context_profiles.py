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


def test_ollama_profile_recognizes_nonstandard_context_tag_suffixes():
    client = core.OllamaClient.__new__(core.OllamaClient)
    client._post = lambda *args, **kwargs: {"parameters": ""}
    client.status = lambda: {"models": []}

    profile = client.model_profile("gemma4:12b-20k")

    assert profile["num_ctx"] == 20 * 1024
    assert profile["source"] == "tag_suffix"
    assert profile["runtime_context_confirmed"] is False


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


def test_llama_cpp_profile_uses_server_props_for_direct_server_without_router_models():
    client = LlamaCppClient.__new__(LlamaCppClient)
    client.model = "bonsai"
    client._available_models = lambda: []
    client._server_json_request = lambda path: {
        "default_generation_settings": {"n_ctx": 65536}
    }

    profile = client.model_profile("bonsai")

    assert profile["num_ctx"] == 65536
    assert profile["runtime_context_limit"] == 65536
    assert "model_context_length" not in profile
    assert profile["runtime_context_confirmed"] is True
    assert profile["source"] == "llama_cpp_runtime_props"


def test_llama_cpp_profile_uses_router_runtime_context_without_architecture_metadata():
    client = LlamaCppClient.__new__(LlamaCppClient)
    client.model = "bonsai"
    client._available_models = lambda: [{
        "name": "bonsai",
        "runtime_context_limit": 65536,
    }]
    client._server_json_request = lambda path: {}

    profile = client.model_profile("bonsai")

    assert profile["num_ctx"] == 65536
    assert profile["runtime_context_limit"] == 65536
    assert "model_context_length" not in profile
    assert profile["runtime_context_confirmed"] is True
    assert profile["source"] == "llama_cpp_router_args"


def test_llama_cpp_router_missing_context_uses_unconfirmed_fallback_without_props():
    client = LlamaCppClient.__new__(LlamaCppClient)
    client.model = "bonsai"
    client._available_models = lambda: [{
        "name": "bonsai",
        "runtime_context_limit": None,
    }]

    def unexpected_props(_path):
        raise AssertionError("Router /props must not be used for a model-specific fallback")

    client._server_json_request = unexpected_props
    profile = client.model_profile("bonsai")

    assert profile["num_ctx"] == core.settings.llama_cpp_context
    assert "model_context_length" not in profile
    assert profile["runtime_context_confirmed"] is False
    assert profile["source"] == "llama_cpp_config_fallback_unconfirmed"


def test_llama_cpp_server_native_url_is_outside_openai_v1_prefix():
    client = LlamaCppClient.__new__(LlamaCppClient)
    client.base = "http://127.0.0.1:8081/v1"

    assert client._server_url("/props") == "http://127.0.0.1:8081/props"
