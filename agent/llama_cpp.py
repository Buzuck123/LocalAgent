"""OpenAI-compatible llama.cpp server adapter.

The adapter deliberately returns the Ollama-shaped response consumed by the
existing agent loop. Provider-specific HTTP and streaming details stay here.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

from . import db
from .config import settings


class LlamaCppClient:
    def __init__(self):
        self.base = settings.llama_cpp_url.rstrip("/")
        self.model = settings.llama_cpp_model
        self._request_lock = threading.RLock()
        self._active_lock = threading.RLock()
        self._active_responses: dict[str, object] = {}
        self.last_generation_stats = {
            "active": False, "tokens_per_second": None, "eval_count": 0,
            "eval_duration": 0, "model": self.model,
        }
        self.opener = build_opener(ProxyHandler({}))

    def _url(self, path: str) -> str:
        return self.base + (path if path.startswith("/") else "/" + path)

    def _json_request(self, path: str, *, method: str = "GET", payload=None, timeout=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = Request(self._url(path), data=data, method=method,
                      headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self.opener.open(req, timeout=timeout or settings.llama_cpp_timeout) as response:
                raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"llama.cpp HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Cannot connect to llama.cpp at {self.base}: {exc}") from exc

    def cancel_generation(self, task_id: str) -> bool:
        with self._active_lock:
            response = self._active_responses.get(str(task_id))
        if response is None:
            return False
        try:
            response.close()
            return True
        except Exception:
            return False

    def model_profile(self, model: str | None = None) -> dict:
        name = model or db.get_app_setting("selected_ollama_model", self.model)
        return {
            "model": name, "num_ctx": int(settings.llama_cpp_context),
            "num_predict": None, "source": "llama_cpp_server_config",
        }

    def _available_models(self) -> list[dict]:
        payload = self._json_request("/models")
        models = []
        for item in payload.get("data", []) if isinstance(payload, dict) else []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append({"name": item["id"], "size": None, "modified_at": None})
        return models

    def models(self) -> dict:
        models = self._available_models()
        names = {item["name"] for item in models}
        selected = db.get_app_setting("selected_ollama_model")
        if selected not in names:
            selected = self.model if self.model in names else (models[0]["name"] if models else "")
            if selected:
                db.set_app_setting("selected_ollama_model", selected)
        return {"models": models, "selected_model": selected}

    def select_model(self, model: str) -> dict:
        name = str(model or "").strip()
        if not name or len(name) > 200:
            raise ValueError("Укажите корректный идентификатор модели llama.cpp")
        available = {item["name"] for item in self._available_models()}
        if name not in available:
            raise ValueError(f"Модель {name!r} не обслуживается текущим llama.cpp сервером")
        db.set_app_setting("selected_ollama_model", name)
        self.model = name
        return {"selected_model": name}

    def status(self) -> dict:
        try:
            from urllib.parse import urlsplit, urlunsplit
            parts = urlsplit(self.base)
            root = urlunsplit((parts.scheme, parts.netloc, parts.path.removesuffix("/v1"), "", ""))
            req = Request(root.rstrip("/") + "/health", headers={"Accept": "application/json"})
            with self.opener.open(req, timeout=10) as response:
                raw = response.read()
            health = json.loads(raw.decode("utf-8")) if raw else {}
            return {"provider": "llama_cpp", "available": True, "health": health}
        except Exception as exc:
            return {"provider": "llama_cpp", "available": False, "error": str(exc)}

    def unload(self) -> dict:
        # llama-server owns model lifetime; it has no Ollama-style unload API.
        return {"ok": True, "unloaded": False,
                "message": "Модель загружена процессом llama-server; остановка выполняется на стороне сервера."}

    @staticmethod
    def _safe_messages(messages: list[dict], model: str) -> list[dict]:
        from .core import _sanitize_messages_for_ollama, _prepare_messages_for_next_model_turn
        sanitized = _prepare_messages_for_next_model_turn(
            _sanitize_messages_for_ollama(messages)
        )
        output = []
        pending_call_ids: list[str] = []
        for source in sanitized:
            role = source.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = source.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            item = {"role": role, "content": content}
            if role == "tool":
                call_id = source.get("tool_call_id")
                if not call_id and pending_call_ids:
                    call_id = pending_call_ids.pop(0)
                if not call_id:
                    # Older checkpoints may not have a provider call id. Such
                    # receipts are retained as user-visible context, not invalid
                    # OpenAI tool messages.
                    item = {"role": "user", "content": "[Предыдущий результат инструмента]\n" + content}
                else:
                    item["tool_call_id"] = call_id
            if role == "assistant" and source.get("tool_calls"):
                calls = []
                for call in source["tool_calls"]:
                    fn = call.get("function") or {}
                    args = fn.get("arguments", {})
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    calls.append({
                        "id": call.get("id") or ("call_" + uuid.uuid4().hex[:20]),
                        "type": "function",
                        "function": {"name": fn.get("name", ""), "arguments": args},
                    })
                if calls:
                    item["tool_calls"] = calls
                    pending_call_ids.extend(str(call["id"]) for call in calls if call.get("id"))
            # The active Bonsai server currently advertises text-only input.
            # Never send Ollama's non-standard images[] field or silently claim
            # that image/audio/video bytes were analysed.
            attachments = source.get("attachments")
            if isinstance(attachments, list) and attachments:
                names = [str(x.get("name") or x.get("kind") or "media")
                         for x in attachments if isinstance(x, dict)]
                warning = (
                    "[Медиа не переданы модели: текущий llama.cpp endpoint объявляет text-only input. "
                    "Файлы: " + ", ".join(names) + ".] "
                    "Продолжай выполнение исходного запроса пользователя."
                )
                item["content"] = warning
                if output and output[-1].get("role") == "tool":
                    try:
                        prior_receipt = json.loads(output[-1].get("content", ""))
                    except (TypeError, json.JSONDecodeError):
                        prior_receipt = None
                    if isinstance(prior_receipt, dict) and isinstance(prior_receipt.get("result"), dict):
                        prior_result = prior_receipt["result"]
                        prior_result["visual_media_delivery"] = "not_attached_llama_cpp_text_mode"
                        prior_note = str(prior_result.get("note") or "").strip()
                        prior_result["note"] = (prior_note + " Визуальные вложения не передавались модели; был передан только извлечённый текст.").strip()
                        output[-1]["content"] = json.dumps(prior_receipt, ensure_ascii=False, default=str)
            if role == "tool" and source.get("tool_name") == "load_file_for_model":
                try:
                    receipt = json.loads(item["content"])
                except (TypeError, json.JSONDecodeError):
                    receipt = None
                if isinstance(receipt, dict) and isinstance(receipt.get("result"), dict):
                    result = receipt["result"]
                    if result.get("kind") in {"image", "audio", "video"}:
                        kind = result["kind"]
                        result["delivery"] = "not_attached_llama_cpp_text_mode"
                        if kind == "video":
                            result["audio_track_handling"] = "not_attached"
                        note = "Текущий llama.cpp сервер объявляет text-only input; медиа-байты не передавались модели."
                        result["note"] = (str(result.get("note") or "").strip() + " " + note).strip()
                        item["content"] = json.dumps(receipt, ensure_ascii=False, default=str)
            output.append(item)
        return output

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             reasoning: bool = True, model: str | None = None) -> dict:
        # Keep the same serialized request semantics as OllamaClient. In
        # particular, cancel_generation closes the active HTTP response.
        with self._request_lock:
            selected = model or db.get_app_setting("selected_ollama_model", self.model)
            model_messages = self._safe_messages(messages, selected)
            payload = {
                "model": selected,
                "messages": model_messages,
                "stream": True,
                            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            task_id = _current_task_id()
            started = time.monotonic()
            request_id = uuid.uuid4().hex
            req = Request(self._url("/chat/completions"), data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            message = {"role": "assistant", "content": ""}
            calls: dict[int, dict] = {}
            usage = {}
            response = None
            self.last_generation_stats = {
                "active": True, "tokens_per_second": 0.0, "estimated": True,
                "eval_count": 0, "model": selected, "started_at": time.time(),
                "request_id": request_id,
            }
            try:
                response = self.opener.open(req, timeout=settings.llama_cpp_timeout)
                if task_id:
                    with self._active_lock:
                        self._active_responses[str(task_id)] = response
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        raise RuntimeError(f"llama.cpp stream error: {str(chunk['error'])[:1500]}")
                    if isinstance(chunk.get("usage"), dict):
                        usage.update(chunk["usage"])
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        message["content"] += str(delta["content"])
                    reasoning_text = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning_text:
                        message.setdefault("thinking", "")
                        message["thinking"] += str(reasoning_text)
                    generated_chars = len(message["content"]) + len(message.get("thinking", ""))
                    elapsed_now = max(0.1, time.monotonic() - started)
                    self.last_generation_stats.update({
                        "active": True, "estimated": True,
                        "tokens_per_second": round((generated_chars / 4) / elapsed_now, 1),
                        "generated_chars": generated_chars,
                        "elapsed_seconds": round(elapsed_now, 1),
                    })
                    for part in delta.get("tool_calls") or []:
                        index = int(part.get("index", 0))
                        current = calls.setdefault(index, {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if part.get("id"):
                            current["id"] = part["id"]
                        fn = part.get("function") or {}
                        if fn.get("name"):
                            current["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            current["function"]["arguments"] += fn["arguments"]
                if calls:
                    normalized = []
                    for index in sorted(calls):
                        call = calls[index]
                        try:
                            call["function"]["arguments"] = json.loads(call["function"]["arguments"] or "{}")
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                f"llama.cpp returned malformed tool arguments for {call['function']['name']!r}"
                            ) from exc
                        normalized.append({
                            "id": call["id"],
                            "type": "function",
                            "function": call["function"],
                        })
                    message["tool_calls"] = normalized
                elapsed = max(0.001, time.monotonic() - started)
                completion_tokens = int(usage.get("completion_tokens") or max(0, len(message["content"]) // 4))
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                self.last_generation_stats = {
                    "active": False, "model": selected, "request_id": request_id,
                    "eval_count": completion_tokens, "prompt_eval_count": prompt_tokens,
                    "tokens_per_second": round(completion_tokens / elapsed, 2) if completion_tokens else None,
                    "elapsed_seconds": round(elapsed, 2), "usage": usage,
                    "num_ctx": int(settings.llama_cpp_context), "num_predict": None,
                }
                return {
                    "model": selected, "message": message,
                    "eval_count": completion_tokens, "prompt_eval_count": prompt_tokens,
                    "eval_duration": int(elapsed * 1_000_000_000),
                    "prompt_eval_duration": 0, "done": True,
                }
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                self.last_generation_stats = {"active": False, "error": detail, "model": selected}
                raise RuntimeError(f"llama.cpp HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                self.last_generation_stats = {"active": False, "error": str(exc), "model": selected}
                raise RuntimeError(f"llama.cpp request failed: {exc}") from exc
            finally:
                if task_id:
                    with self._active_lock:
                        if self._active_responses.get(str(task_id)) is response:
                            self._active_responses.pop(str(task_id), None)
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass


def _current_task_id():
    # Import lazily to avoid the core <-> provider module import cycle.
    from .core import _CURRENT_MODEL_TASK_ID
    return _CURRENT_MODEL_TASK_ID.get()
