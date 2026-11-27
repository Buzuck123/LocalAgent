"""Compact, model-facing terminal diagnostic summary.

The controller owns operation polling and recovery order. This module only
formats observations; it does not authorize actions or maintain evidence IDs.
"""
from __future__ import annotations

import json
from typing import Any


def _container_state(snapshot: dict[str, Any]) -> str:
    item = snapshot.get("container_inspect")
    if not isinstance(item, dict) or item.get("returncode") != 0:
        return "не удалось определить"
    try:
        payload = json.loads(item.get("output") or "[]")
        state = payload[0].get("State") if isinstance(payload, list) and payload else {}
        if not isinstance(state, dict):
            return "не удалось определить"
        return "работает" if state.get("Running") else "остановлен"
    except (TypeError, ValueError, IndexError, AttributeError):
        return "не удалось определить"


def build_diagnostic_context(
    snapshot: dict[str, Any],
    previous_evidence: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a compact diagnostic report; previous_evidence is accepted for compatibility."""
    command = str(snapshot.get("command") or "")
    output = str(snapshot.get("terminal_output_tail") or "")
    report = {
        "command": command,
        "operation": {
            "id": str(snapshot.get("operation_id") or ""),
            "state": str(snapshot.get("operation_state") or "unknown"),
            "elapsed_seconds": round(float(snapshot.get("elapsed_seconds") or 0), 1),
            "silent_seconds": round(float(snapshot.get("idle_seconds") or 0), 1),
            "exit_code": snapshot.get("operation_exit_code"),
            "last_output": output[-2500:] or "(вывода нет)",
            "runtime_error": str(snapshot.get("runtime_error") or snapshot.get("operation_poll_error") or "")[:500],
        },
        "runtime": {
            "swerex_responsive": snapshot.get("runtime_responsive") is True,
            "container": _container_state(snapshot),
        },
        "important_notes": [
            "Отсутствие вывода само по себе не доказывает зависание.",
            "Список процессов и ресурсы относятся ко всему контейнеру, а не обязательно к исходной операции.",
            "Проверка SWE-ReX подтверждает доступность API, но не полезный прогресс команды.",
            "Основную Bash-сессию нельзя безопасно проверять параллельно с выполняющейся в ней операцией.",
        ],
    }
    return report, []
