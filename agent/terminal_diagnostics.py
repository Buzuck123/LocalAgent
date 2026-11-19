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
                     "Исходная команда сохранена полностью в поле operation.command.",
                     "Это команда, выполняющаяся в основной Bash-сессии.",
                     "Сама команда не говорит, зависла ли операция.")
    evidence[-1]["command"] = command[:100000]

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
    if snapshot.get("operation_poll_error"):
        add("operation_state_unavailable", "swerex_get_operation", "operation",
            "Не удалось обновить состояние самой операции: " + str(snapshot.get("operation_poll_error")),
            "Контроллер не смог получить свежий статус SWE-ReX operation.",
            "При неизвестном статусе прерывание не разрешается.")
    if snapshot.get("runtime_error"):
        add("runtime_error", "terminal_monitor", "operation",
            "Последняя ошибка runtime: " + str(snapshot.get("runtime_error"))[:300],
            "Может указывать на проблему взаимодействия с runtime.",
            "Ошибка опроса не доказывает, что удалённый процесс остановился.")
    poll_errors = snapshot.get("runtime_poll_errors") or []
    if poll_errors:
        add("runtime_poll_errors", "terminal_monitor", "runtime",
            "В предыдущем цикле были ошибки опроса runtime (" + str(len(poll_errors)) + ").",
            "Показывает нестабильность канала опроса.",
            "Не устанавливает состояние самой команды.")

    idle_id = add(
        "no_terminal_output", "terminal_monitor", "operation",
        f"Нового вывода нет {idle:.1f} с; общее время выполнения {elapsed:.1f} с.",
        "Показывает длительность тишины в основной PTY.",
        "Тихая команда может штатно ожидать сеть, ввод или завершение фоновой работы.",
    )
    evidence[-1]["value_seconds"] = idle
    if snapshot.get("new_output_since_cursor"):
        add(
            "new_terminal_output", "swerex_get_operation", "operation",
            "SWE-ReX вернул новый фрагмент вывода после предыдущего курсора.",
            "Есть новое наблюдаемое действие команды.",
            "Сам по себе вывод не гарантирует корректное завершение операции.",
        )

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

    stats = snapshot.get("container_stats")
    if isinstance(stats, dict) and stats.get("returncode") == 0:
        fields = str(stats.get("output") or "").strip().split("|")
        if len(fields) >= 4:
            observation = (
                f"Контейнер: CPU {fields[0]}, память {fields[1]} ({fields[2]}), процессов {fields[3]}."
            )
        else:
            observation = "Получены агрегированные метрики CPU и памяти контейнера."
        add("environment_metric", "podman_stats", "container", observation,
            "Показывает ресурсы всего контейнера, а не отдельной terminal operation.",
            "Низкая или высокая загрузка контейнера не доказывает зависание конкретной команды.")

    memory = snapshot.get("host_memory")
    if isinstance(memory, dict) and memory.get("returncode") == 0:
        rows = [line.split() for line in str(memory.get("output") or "").splitlines() if line.strip()]
        mem_row = next((row for row in rows if row and row[0] == "Mem:"), None)
        if mem_row and len(mem_row) >= 7:
            observation = f"Память хоста: всего {mem_row[1]}, занято {mem_row[2]}, доступно {mem_row[6]}."
        else:
            observation = "Состояние памяти хоста получено, но не удалось компактно разобрать вывод."
        add("environment_metric", "host_memory", "host", observation,
            "Показывает общую доступность памяти хоста.",
            "Не устанавливает, какая именно команда потребляет память.")

    disk = snapshot.get("host_disk")
    if isinstance(disk, dict) and disk.get("returncode") == 0:
        rows = [line.split() for line in str(disk.get("output") or "").splitlines() if line.strip()]
        row = rows[-1] if len(rows) > 1 else []
        observation = (
            f"Файловая система workspace: размер {row[1]}, занято {row[2]}, свободно {row[3]}, заполнение {row[4]}."
            if len(row) >= 5 else "Состояние диска workspace получено, но не удалось разобрать вывод."
        )
        add("environment_metric", "host_disk", "host", observation,
            "Показывает заполнение файловой системы, содержащей workspace.",
            "Не объясняет зависание, если файловая система не заполнена.")

    load = snapshot.get("host_load")
    if isinstance(load, dict) and load.get("returncode") == 0:
        add("environment_metric", "host_load", "host",
            "Снята общая нагрузка хоста: " + str(load.get("output") or "").strip()[:180],
            "Показывает общую нагрузку хоста.",
            "Load average не устанавливает причину задержки конкретной команды.")

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
            "command": command[:100000],
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
    from datetime import datetime, timezone
    try:
        sampled = datetime.fromisoformat(str(snapshot.get("sampled_at") or "").replace("Z", "+00:00"))
        if sampled.tzinfo is None:
            sampled = sampled.replace(tzinfo=timezone.utc)
        for item in selected:
            observed = datetime.fromisoformat(str(item.get("timestamp") or "").replace("Z", "+00:00"))
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=timezone.utc)
            if abs((sampled - observed).total_seconds()) > 180:
                return False, "Одно из доказательств устарело; требуется новый диагностический снимок."
    except (TypeError, ValueError):
        return False, "Не удалось проверить время получения доказательств."
    if action == "interrupt_terminal_operation":
        kinds = {item.get("kind") for item in selected}
        state = str(snapshot.get("operation_state") or "unknown")
        elapsed = float(snapshot.get("elapsed_seconds") or 0)
        hard_timeout = snapshot.get("command_hard_timeout_seconds")
        operation_is_running = state in {"running", "waiting_input"}
        timeout_elapsed = (
            operation_is_running
            and isinstance(hard_timeout, (int, float)) and hard_timeout > 0
            and elapsed >= float(hard_timeout)
            and "operation_state" in kinds
        )
        process_inspection = any(
            item.get("kind") == "diagnostic_terminal"
            and item.get("success") is True
            and str(item.get("command") or "").split(" ", 1)[0] in {"ps", "lsof"}
            for item in selected
        )
        repeated_silence = (
            state in {"running", "waiting_input"}
            and elapsed >= 900
            and float(snapshot.get("idle_seconds") or 0) >= 900
            and "operation_state" in kinds
            and "no_terminal_output" in kinds
            and "repeated_no_progress" in kinds
            and process_inspection
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
