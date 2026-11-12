import io
import json

from agent import core


def test_model_profile_uses_exact_modelfile_num_ctx(monkeypatch):
    client = core.OllamaClient()
    monkeypatch.setattr(core.db, "get_app_setting", lambda key, default=None: default)
    monkeypatch.setattr(core.db, "set_app_setting", lambda key, value: None)
    calls = []

    def fake_post(self, endpoint, payload, timeout=None):
        calls.append((endpoint, payload))
        return {"parameters": "temperature 0.7\nnum_ctx 32768\nnum_predict 4096"}

    monkeypatch.setattr(core.OllamaClient, "_post", fake_post)
    profile = client.model_profile("qwen3.5:9b-32k")
    assert calls[0][0] == "/api/show"
    assert calls[0][1]["model"] == "qwen3.5:9b-32k"
    assert profile["num_ctx"] == 32768
    assert profile["num_predict"] == 4096
    assert profile["source"] == "modelfile"


def test_model_profile_uses_context_suffix_when_modelfile_has_no_num_ctx(monkeypatch):
    client = core.OllamaClient()
    monkeypatch.setattr(core.db, "get_app_setting", lambda key, default=None: default)
    monkeypatch.setattr(core.db, "set_app_setting", lambda key, value: None)
    monkeypatch.setattr(core.OllamaClient, "_post", lambda self, *a, **k: {"parameters": "temperature 0.7"})
    profile = client.model_profile("qwen3.5:9b-32k")
    assert profile["num_ctx"] == 32768
    assert profile["num_predict"] is None
    assert profile["source"] == "tag_suffix"


def test_chat_uses_pinned_model_context_and_streams_live_stats(monkeypatch):
    client = core.OllamaClient()
    monkeypatch.setattr(core.db, "get_app_setting", lambda key, default=None: default)
    monkeypatch.setattr(core.db, "set_app_setting", lambda key, value: None)
    monkeypatch.setattr(core.OllamaClient, "_post", lambda self, *a, **k: {"parameters": "num_ctx 32768"})
    captured = {}

    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *args): self.close()

    def open_request(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        lines = [
            {"message": {"role": "assistant", "content": "hello "}, "done": False},
            {"message": {"role": "assistant", "content": "world"}, "done": False},
            {"message": {}, "done": True, "model": "qwen3.5:9b-32k", "eval_count": 4,
             "eval_duration": 1_000_000_000, "prompt_eval_count": 20,
             "prompt_eval_duration": 2_000_000_000},
        ]
        return Response(("\n".join(json.dumps(x) for x in lines) + "\n").encode())

    client.opener = type("Opener", (), {"open": staticmethod(open_request)})()
    token = core._CURRENT_MODEL_OVERRIDE.set("qwen3.5:9b-32k")
    context_token = core._CURRENT_MODEL_CONTEXT.set(32768)
    try:
        result = client.chat([{"role": "user", "content": "test"}], reasoning=False)
    finally:
        core._CURRENT_MODEL_CONTEXT.reset(context_token)
        core._CURRENT_MODEL_OVERRIDE.reset(token)

    payload = captured["payload"]
    assert payload["stream"] is True
    # The tag's Modelfile owns runtime parameters; LocalAgent must not
    # override them with a per-request options object.
    assert "options" not in payload
    assert result["message"]["content"] == "hello world"
    assert client.last_generation_stats["tokens_per_second"] == 4.0
    assert client.last_generation_stats["prompt_tokens_per_second"] == 10.0
    assert client.last_generation_stats["active"] is False
