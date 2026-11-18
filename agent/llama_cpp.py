"""OpenAI-compatible llama.cpp server adapter.

The adapter deliberately returns the Ollama-shaped response consumed by the
existing agent loop. Provider-specific HTTP and streaming details stay here.
"""
from __future__ import annotations

import base64
import mimetypes
import json
from pathlib import Path
import threading
import time
import uuid
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
        self._recovery_required = False
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
        name = model or db.get_app_setting("selected_llama_cpp_model", self.model)
        context = int(settings.llama_cpp_context)
        try:
            item = next((x for x in self._available_models() if x["name"] == name), None)
            if item and item.get("context_length"):
                context = int(item["context_length"])
        except Exception:
            pass
        return {
            "model": name, "num_ctx": context,
            "num_predict": None, "source": "llama_cpp_server_config",
        }

    def _available_models(self) -> list[dict]:
        payload = self._json_request("/models")
        models = []
        for item in payload.get("data", []) if isinstance(payload, dict) else []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
                models.append({
                    "name": item["id"], "size": None, "modified_at": None,
                    "context_length": item.get("context_length") or meta.get("n_ctx_train"),
                })
        return models

    def models(self) -> dict:
        models = self._available_models()
        names = {item["name"] for item in models}
        selected = db.get_app_setting("selected_llama_cpp_model")
        if selected not in names:
            selected = self.model if self.model in names else (models[0]["name"] if models else "")
            if selected:
                db.set_app_setting("selected_llama_cpp_model", selected)
        return {"models": models, "selected_model": selected}

    def select_model(self, model: str) -> dict:
        name = str(model or "").strip()
        if not name or len(name) > 200:
            raise ValueError("Укажите корректный идентификатор модели llama.cpp")
        available = {item["name"] for item in self._available_models()}
        if name not in available:
            raise ValueError(f"Модель {name!r} не обслуживается текущим llama.cpp сервером")
        db.set_app_setting("selected_llama_cpp_model", name)
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

    def _recover_after_timeout(self) -> None:
        if not self._recovery_required:
            return
        status = self.status()
        if not status.get("available"):
            raise RuntimeError(
                "llama.cpp did not pass its health check after an interrupted stream; "
                "the previous request was not replayed. Check the server before continuing."
            )
        self._recovery_required = False

    def unload(self) -> dict:
        # llama-server owns model lifetime; it has no Ollama-style unload API.
        return {"ok": True, "unloaded": False,
                "message": "Модель загружена процессом llama-server; остановка выполняется на стороне сервера."}

    @staticmethod
    def _media_parts(attachments: list[dict], text: str) -> list[dict]:
        """Convert managed media files to OpenAI-compatible content parts."""
        parts: list[dict] = [{"type": "text", "text": text}] if text else []

        def read_managed(path_value: object) -> bytes:
            path = Path(str(path_value or "")).resolve(strict=True)
            if not path.is_file():
                raise ValueError("Мультимодальное вложение не является обычным файлом")
            size = path.stat().st_size
            if size <= 0 or size > 100 * 1024 * 1024:
                raise ValueError("Мультимодальное вложение пустое или превышает 100 МБ")
            return path.read_bytes()

        def add_image(path: str, label: str = "") -> None:
            raw = read_managed(path)
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                mime = "image/jpeg"
            if label:
                parts.append({"type": "text", "text": label})
            encoded = base64.b64encode(raw).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})

        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            kind = str(attachment.get("kind") or "")
            path = str(attachment.get("path") or "")
            name = str(attachment.get("name") or kind or "media")
            if kind == "image":
                add_image(path, f"Прикреплённое изображение: {name}")
            elif kind == "audio":
                raw = read_managed(path)
                encoded = base64.b64encode(raw).decode("ascii")
                parts.append({"type": "text", "text": f"Аудиовложение: {name}. Прослушай и проанализируй звук."})
                parts.append({"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}})
            elif kind == "video":
                from .attachments import extract_video_frames, extract_video_audio, ATTACHMENT_ROOT
                sampled = extract_video_frames(path, max_frames=60)
                parts.append({"type": "text", "text": f"Видео: {name}, длительность {sampled['duration_seconds']} с. Кадры передаются по порядку с временными метками."})
                root = ATTACHMENT_ROOT.resolve()
                for index, frame in enumerate(sampled.get("frames", []), 1):
                    frame_path = Path(frame["path"]).resolve(strict=True)
                    if frame_path == root or root not in frame_path.parents:
                        raise ValueError("Извлечённый кадр видео находится вне каталога вложений")
                    add_image(str(frame_path), f"Кадр {index}, {frame['timestamp_seconds']:.2f} с")
                audio = extract_video_audio(path, max_seconds=30.0)
                if audio:
                    audio_path = Path(audio["path"]).resolve(strict=True)
                    if audio_path == root or root not in audio_path.parents:
                        raise ValueError("Извлечённая звуковая дорожка находится вне каталога вложений")
                    encoded = base64.b64encode(read_managed(audio_path)).decode("ascii")
                    parts.append({"type": "text", "text": "Звуковая дорожка видео (первые 30 секунд):"})
                    parts.append({"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}})
        return parts

    @staticmethod
    def _safe_messages(messages: list[dict], model: str) -> list[dict]:
        from .core import _sanitize_messages_for_ollama, _prepare_messages_for_next_model_turn
        sanitized = _prepare_messages_for_next_model_turn(
            _sanitize_messages_for_ollama(messages)
        )
        output = []
        system_parts: list[str] = []
        pending_call_ids: list[str] = []
        for source in sanitized:
            role = source.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = source.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            item = {"role": role, "content": content}
            if role == "tool":
                call_id = source.get("tool_call_id")
                if not call_id and pending_call_ids:
                    call_id = pending_call_ids.pop(0)
                if not call_id:
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
            attachments = source.get("attachments")
            if role == "user" and isinstance(attachments, list) and attachments:
                item["content"] = LlamaCppClient._media_parts(attachments, content)
                if output and output[-1].get("role") == "tool":
                    try:
                        prior_receipt = json.loads(output[-1].get("content", ""))
                    except (TypeError, json.JSONDecodeError):
                        prior_receipt = None
                    if isinstance(prior_receipt, dict) and isinstance(prior_receipt.get("result"), dict):
                        prior_receipt["result"]["visual_media_delivery"] = "attached_to_llama_cpp_multimodal_request"
                        note = str(prior_receipt["result"].get("note") or "").strip()
                        prior_receipt["result"]["note"] = (note + " Визуальные вложения передаются следующему запросу через мультимодальный API llama.cpp.").strip()
                        output[-1]["content"] = json.dumps(prior_receipt, ensure_ascii=False, default=str)
            if role == "tool" and source.get("tool_name") == "load_file_for_model":
                try:
                    receipt = json.loads(item["content"])
                except (TypeError, json.JSONDecodeError):
                    receipt = None
                if isinstance(receipt, dict) and isinstance(receipt.get("result"), dict):
                    result = receipt["result"]
                    if result.get("kind") in {"image", "audio", "video"}:
                        result["delivery"] = "attached_to_llama_cpp_multimodal_request"
                        result["note"] = "Медиафайл будет передан в мультимодальном запросе следующего хода."
                        item["content"] = json.dumps(receipt, ensure_ascii=False, default=str)
            output.append(item)
        if system_parts:
            output.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        return output

    @staticmethod
    def _trace_safe_messages(messages: list[dict]) -> list[dict]:
        safe = []
        for message in messages:
            item = dict(message)
            content = item.get("content")
            if isinstance(content, list):
                item["content"] = [
                    ({"type": part.get("type"), "text": part.get("text", "[media payload omitted]")}
                     if part.get("type") == "text" else {"type": part.get("type"), "payload": "[omitted]"})
                    for part in content if isinstance(part, dict)
                ]
            safe.append(item)
        return safe

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             reasoning: bool = True, model: str | None = None) -> dict:
        # Keep the same serialized request semantics as OllamaClient. In
        # particular, cancel_generation closes the active HTTP response.
        with self._request_lock:
            self._recover_after_timeout()
            selected = model or db.get_app_setting("selected_llama_cpp_model", self.model)
            model_messages = self._safe_messages(messages, selected)
            payload = {
                "model": selected,
                "messages": model_messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            task_id = _current_task_id()
            from .core import _CURRENT_MODEL_CONTEXT
            context_limit = _CURRENT_MODEL_CONTEXT.get() or settings.llama_cpp_context
            from .core import estimate_tokens
            try:
                estimated_prompt_tokens = int(estimate_tokens(model_messages, model=selected))
            except Exception:
                estimated_prompt_tokens = 0
            db.record_debug_trace(task_id, "llm_request", {
                "model": selected,
                "reasoning_requested": bool(reasoning),
                "model_profile": self.model_profile(selected),
                "generation_parameters": {"stream": True, "request_options_sent": False},
                "messages": self._trace_safe_messages(model_messages),
                "tools": tools or [],
                "context_estimate": context_limit,
                "provider": "llama_cpp",
            })
            started = time.monotonic()
            request_id = uuid.uuid4().hex
            req = Request(self._url("/chat/completions"), data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            message = {"role": "assistant", "content": ""}
            calls: dict[int, dict] = {}
            usage = {}
            response = None
            stream_done = False
            self.last_generation_stats = {
                "active": True, "tokens_per_second": 0.0, "estimated": True,
                "eval_count": 0, "prompt_eval_count": 0,
                "current_prompt_tokens": estimated_prompt_tokens,
                "context_used_tokens": estimated_prompt_tokens, "num_ctx": int(context_limit),
                "model": selected, "started_at": time.time(),
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
                        stream_done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        raise RuntimeError(f"llama.cpp stream error: {str(chunk['error'])[:1500]}")
                    if isinstance(chunk.get("usage"), dict):
                        usage.update(chunk["usage"])
                        prompt_now = int(usage.get("prompt_tokens") or 0)
                        self.last_generation_stats.update({
                            "prompt_eval_count": prompt_now,
                            "current_prompt_tokens": prompt_now,
                        })
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
                        "current_prompt_tokens": int(usage.get("prompt_tokens") or estimated_prompt_tokens),
                        "context_used_tokens": int(usage.get("prompt_tokens") or estimated_prompt_tokens),
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
                if not stream_done:
                    raise RuntimeError("llama.cpp SSE stream ended before the [DONE] marker")
                if calls:
                    normalized = []
                    for index in sorted(calls):
                        call = calls[index]
                        if not call.get("id"):
                            call["id"] = "call_" + uuid.uuid4().hex[:20]
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
                prompt_tokens = int(usage.get("prompt_tokens") or estimated_prompt_tokens)
                self.last_generation_stats = {
                    "active": False, "model": selected, "request_id": request_id,
                    "eval_count": completion_tokens, "prompt_eval_count": prompt_tokens,
                    "tokens_per_second": round(completion_tokens / elapsed, 2) if completion_tokens else None,
                    "elapsed_seconds": round(elapsed, 2), "usage": usage,
                    "num_ctx": int(context_limit), "num_predict": None,
                    "current_prompt_tokens": prompt_tokens,
                    "context_used_tokens": prompt_tokens,
                    "prompt_tokens_per_second": None,
                }
                result = {
                    "model": selected, "message": message,
                    "eval_count": completion_tokens, "prompt_eval_count": prompt_tokens,
                    "eval_duration": int(elapsed * 1_000_000_000),
                    "prompt_eval_duration": 0, "done": True,
                }
                db.record_debug_trace(task_id, "llm_response", {
                    "model": selected, "message": message,
                    "usage": usage, "provider": "llama_cpp",
                })
                if message.get("thinking") and db._debug_setting_enabled(
                    "debug_llm_trace", getattr(settings, "debug_llm_trace", False)
                ) and db._debug_setting_enabled(
                    "debug_llm_thinking", getattr(settings, "debug_llm_thinking", False)
                ):
                    db.record_debug_trace(task_id, "llm_thinking", {
                        "model": selected, "thinking": message["thinking"], "provider": "llama_cpp",
                    }, thinking=True)
                return result
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                self.last_generation_stats = {"active": False, "error": detail, "model": selected}
                raise RuntimeError(f"llama.cpp HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                self.last_generation_stats = {"active": False, "error": str(exc), "model": selected}
                if response is not None:
                    self._recovery_required = True
                raise RuntimeError(f"llama.cpp request failed: {exc}") from exc
            except Exception as exc:
                self.last_generation_stats = {"active": False, "error": str(exc)[:2000], "model": selected}
                if response is not None and not isinstance(exc, HTTPError):
                    self._recovery_required = True
                raise
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
