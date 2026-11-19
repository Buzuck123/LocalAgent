"""Compact terminal diagnostic facts and conservative recovery authorization.

Raw host/container metrics stay inside the collector. This module turns them into
small, timestamped observations and applies deterministic gates before recovery.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any


def _evidence_id() -> str:
    return "ev_" + uuid.uuid4().hex[:12]


def _parse_inspect(snapshot: dict[str, Any]) -> dict[str, Any]:
    item = snapshot.get("container_inspect")
    if not isinstance(item, dict) or item.get("returncode") != 0:
        return {}
    try:
        payload = json.loads(item.get("output") or "[]")
        return payload[0] if isinstance(payload, list) and payload else {}
    except (TypeError, ValueError, IndexError):
        return {}


def build_diagnostic_context(snapshot: dict[str, Any],
                             previous_evidence: list[dict[str, Any]] | None = None
                             ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a concise model-facing report and newly collected evidence records."""
    previous = [x for x in (previous_evidence or []) if isinstance(x, dict)]
    evidence: list[dict[str, Any]] = []
    sampled_at = str(snapshot.get("sampled_at") or datetime.now(timezone.utc).isoformat())
    operation_id = str(snapshot.get("operation_id") or "")

    def add(kind: str, source: str, scope: str, observation: str,
            meaning: str, limitation: str = "") -> str:
        item = {
            "id": _evidence_id(), "kind": kind, "source": source,
            "timestamp": sampled_at, "scope": scope,
            "operation_id": operation_id, "observation": observation,
            "meaning": meaning, "limitation": limitation,
        }
        evidence.append(item)
        return item["id"]

    command = str(snapshot.get("command") or "")
    command_id = add("command", "terminal_operation", "operation",
                     command[:8000] or "Исходная команда не сохранена.",
                     "Это команда, выполняющаяся в основной Bash-сессии.",
                     "Сама команда не говорит, зависла ли операция.")

    elapsed = float(snapshot.get("elapsed_seconds") or 0)
    idle = float(snapshot.get("idle_seconds") or 0)
    state = str(snapshot.get("operation_state") or "unknown")
    runtime_ok = snapshot.get("runtime_responsive") is True
    runtime_id = add("runtime_health", "swerex_health", "runtime",
                     "SWE-ReX отвечает." if runtime_ok else "SWE-ReX не подтвердил доступность.",
                     "Проверяет доступность API runtime.",
                     "Доступность API не доказывает прогресс конкретной команды.")

    operation_id_evidence = add(
        "operation_state", "swerex_get_operation", "operation",
        f"Состояние операции: {state}.",
        "Это состояние самой SWE-ReX operation, а не контейнера.",
        "Состояние running не гарантирует, что команда делает полезную работу.",
    ) if state != "unknown" else None

    idle_id = add(
        "no_terminal_output", "terminal_monitor", "operation",
        f"Нового вывода нет {idle:.1f} с; общее время выполнения {elapsed:.1f} с.",
        "Показывает длительность тишины в основной PTY.",
        "Тихая команда может штатно ожидать сеть, ввод или завершение фоновой работы.",
    )
    evidence[-1]["value_seconds"] = idle

    inspect = _parse_inspect(snapshot)
    container_state = (inspect.get("State") or {}) if isinstance(inspect, dict) else {}
    if container_state:
        running = bool(container_state.get("Running"))
        add("container_state", "podman_inspect", "container",
            "Контейнер работает." if running else "Контейнер остановлен.",
            "Подтверждает состояние контейнера на момент проверки.",
            "Не показывает состояние отдельной команды.")
    else:
        add("container_state", "podman_inspect", "container",
            "Состояние контейнера не удалось подтвердить.",
            "Данных недостаточно для вывода о контейнере.")

    # Container-wide process/resource metrics are deliberately summarized,
    # never presented as operation-specific process evidence.
    top = snapshot.get("container_top")
    if isinstance(top, dict) and top.get("returncode") == 0:
        add("container_processes", "podman_top", "container",
            "Получен список процессов контейнера.",
            "Помогает понять, какие процессы существуют в контейнере.",
            "Это общий список контейнера; принадлежность PID конкретной operation не доказана.")

    for key, label, explanation in (
        ("container_stats", "Ресурсы контейнера", "CPU/RAM — агрегаты всего контейнера, не отдельной команды."),
        ("host_memory", "Память хоста", "Показывает общую доступность памяти на хосте."),
        ("host_disk", "Диск workspace", "Показывает заполнение файловой системы, содержащей workspace."),
        ("host_load", "Нагрузка хоста", "Общая нагрузка хоста не устанавливает причину задержки команды."),
    ):
        value = snapshot.get(key)
        if isinstance(value, dict) and value.get("error"):
            add("environment_metric", key, "host" if key.startswith("host_") else "container",
                f"{label}: проверка не удалась ({value.get('error')}).",
                explanation)
        elif isinstance(value, dict):
            raw = str(value.get("output") or "").strip()
            if raw:
                add("environment_metric", key, "host" if key.startswith("host_") else "container",
                    f"{label}: {raw[:700]}", explanation)

    # Compare only operation-scoped facts from distinct samples. A stable
    # no-output interval is evidence of silence, not proof of a deadlock.
    prior_samples = [
        x for x in previous
        if x.get("kind") == "no_terminal_output"
        and x.get("operation_id") == operation_id
    ]
    no_progress_id = None
    if prior_samples and state in {"running", "waiting_input"}:
        from datetime import datetime
        current_time = datetime.fromisoformat(sampled_at.replace("Z", "+00:00"))
        comparable = False
        for prior in reversed(prior_samples):
            try:
                prior_time = datetime.fromisoformat(str(prior.get("timestamp") or "").replace("Z", "+00:00"))
                seconds_between = (current_time - prior_time).total_seconds()
                prior_idle = float(prior.get("value_seconds") or 0)
                if seconds_between >= 15 and idle >= prior_idle + 10:
                    comparable = True
                    break
            except (TypeError, ValueError):
                continue
        if comparable:
            no_progress_id = add(
                "repeated_no_progress", "diagnostic_comparison", "operation",
                "Операция всё ещё числится выполняющейся; между снимками прошло не менее 15 секунд, нового вывода не было.",
                "Подтверждает отсутствие наблюдаемого прогресса по выводу в нескольких разнесённых во времени снимках.",
                "Не исключает штатное ожидание ввода, сети, I/O или внешнего события.",
            )

    report = {
        "operation": {
            "command": command[:8000],
            "elapsed_seconds": round(elapsed, 1),
            "silent_seconds": round(idle, 1),
            "runtime_state": state,
            "runtime_responsive": runtime_ok,
            "last_output": str(snapshot.get("terminal_output_tail") or "")[-1500:] or "(вывода нет)",
        },
        "observations": [
            {"id": x["id"], "fact": x["observation"], "meaning": x["meaning"],
             **({"limitation": x["limitation"]} if x.get("limitation") else {})}
            for x in evidence
        ],
        "important_limits": [
            "Тишина в терминале не доказывает зависание.",
            "CPU/RAM и container_top относятся ко всему контейнеру, а не к конкретной operation.",
            "SWE-ReX health check подтверждает доступность API, а не прогресс команды.",
        ],
        "evidence_ids": [x["id"] for x in evidence],
        "command_evidence_id": command_id,
        "runtime_evidence_id": runtime_id,
        "operation_state_evidence_id": operation_id_evidence,
        "silence_evidence_id": idle_id,
        "repeated_no_progress_evidence_id": no_progress_id,
    }
    return report, evidence


def authorize_recovery(action: str, proposal: dict[str, Any], evidence: list[dict[str, Any]],
                       snapshot: dict[str, Any]) -> tuple[bool, str]:
    """Validate LLM evidence references and enforce conservative controller gates."""
    by_id = {str(item.get("id")): item for item in evidence if isinstance(item, dict)}
    references = proposal.get("evidence_ids")
    if not isinstance(references, list) or not references:
        return False, "Модель не указала идентификаторы доказательств."
    selected = [by_id.get(str(item)) for item in references]
    if any(item is None for item in selected):
        return False, "Предложение ссылается на отсутствующие доказательства."
    operation_id = str(snapshot.get("operation_id") or "")
    if any(item.get("operation_id") != operation_id for item in selected):
        return False, "Доказательства относятся к другой операции."
    if action == "interrupt_terminal_operation":
        kinds = {item.get("kind") for item in selected}
        state = str(snapshot.get("operation_state") or "unknown")
        elapsed = float(snapshot.get("elapsed_seconds") or 0)
        hard_timeout = snapshot.get("command_hard_timeout_seconds")
        timeout_elapsed = (
            isinstance(hard_timeout, (int, float)) and hard_timeout > 0
            and elapsed >= float(hard_timeout)
        )
        repeated_silence = (
            state in {"running", "waiting_input"}
            and elapsed >= 900
            and float(snapshot.get("idle_seconds") or 0) >= 900
            and "no_terminal_output" in kinds
            and "repeated_no_progress" in kinds
            and "diagnostic_terminal" in kinds
        )
        if timeout_elapsed or repeated_silence:
            return True, "Подтверждены hard timeout или длительное отсутствие наблюдаемого прогресса в повторных снимках."
        return False, "Недостаточно оснований: нужен достигнутый hard timeout либо повторно подтверждённая тишина при длительно выполняющейся операции."
    if action == "reset_bash_session":
        if str(snapshot.get("operation_state") or "") in {"running", "waiting_input"}:
            return False, "Нельзя сбрасывать Bash-сессию, пока основная операция ещё выполняется."
        return False, "Сброс Bash разрешается контроллером только после проверки завершения операции и неудачной проверки основной сессии."
    if action == "restart_sandbox_container":
        return False, "Перезапуск контейнера не разрешается по одному предложению LLM; требуется отдельная проверка runtime и состояния контейнера."
    return False, "Неизвестное восстановительное действие."
