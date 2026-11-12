"""Persistent task-plan structure and status transitions."""
from __future__ import annotations

import json
import re
from typing import Any
from pathlib import Path


PLAN_VERSION = 1
STEP_STATUSES = {"pending", "in_progress", "completed", "skipped", "question"}
PLAN_STATUSES = {"in_progress", "completed", "paused", "revision_needed"}


def _valid_acceptance_check(check: Any) -> bool:
    """Validate a deterministic, machine-checkable acceptance criterion."""
    if not isinstance(check, dict):
        return False

    kind = check.get("type")
    if kind in {"file_exists", "directory_exists"}:
        return (
            set(check) == {"type", "path"}
            and isinstance(check.get("path"), str)
            and bool(check["path"].strip())
        )

    if kind == "file_contains":
        return (
            set(check) == {"type", "path", "text"}
            and isinstance(check.get("path"), str)
            and bool(check["path"].strip())
            and isinstance(check.get("text"), str)
            and bool(check["text"])
        )

    if kind == "python_code_exit_code":
        return set(check) == {"type"}

    if kind in {"python_exit_code", "python_tests_passed"}:
        return (
            set(check) == {"type", "path"}
            and isinstance(check.get("path"), str)
            and bool(check["path"].strip())
        )

    return False


def _acceptance_path_candidate(path: str, workspace_root: str | Path, task_id: str | None = None) -> Path:
    """Resolve workspace-relative paths or explicit /work paths without escaping their roots."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("acceptance_check.path must be a non-empty path")
    raw = path.strip().replace("\\", "/")
    root = Path(workspace_root).resolve()
    if raw == "/workspace":
        return root
    if raw.startswith("/workspace/"):
        raw = raw[len("/workspace/"):]
    if raw.startswith("/work/"):
        if task_id is None:
            raise ValueError("/work acceptance checks require the current task context")
        suffix = raw[len("/work/"):]
        # Reuse the sandbox's symlink and traversal protections. Import locally to
        # avoid coupling the planning module to tool initialization at import time.
        from .tools import _relative_file_under, _task_work_directory
        work_dir, _ = _task_work_directory(root, task_id)
        return _relative_file_under(work_dir, suffix, "acceptance_check.path")
    relative = Path(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("acceptance_check.path must be a safe relative workspace path or /work path")
    candidate = (root / relative).resolve(strict=True)
    candidate.relative_to(root)
    return candidate


def verify_acceptance_check(
    check: Any,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    workspace_root: str | Path,
    task_id: str | None = None,
) -> str | None:
    """Verify a criterion from terminal output or a legacy checkpoint receipt."""
    if not _valid_acceptance_check(check) or not isinstance(result, dict):
        return None
    kind, path = check["type"], check.get("path")

    if tool_name == "terminal":
        if result.get("exit_code") != 0 or result.get("timed_out") is True:
            return None
        import shlex
        command = arguments.get("command")
        if not isinstance(command, str):
            command = result.get("command")
        if not isinstance(command, str):
            return None
        try:
            parts = shlex.split(command)
        except ValueError:
            return None
        if kind == "file_exists" and len(parts) == 3 and parts[:2] == ["test", "-f"] and parts[2] == path:
            return f"Фактически проверено в Linux-среде: файл существует — {path}"
        if kind == "directory_exists" and len(parts) == 3 and parts[:2] == ["test", "-d"] and parts[2] == path:
            return f"Фактически проверено в Linux-среде: каталог существует — {path}"
        if kind == "file_contains" and len(parts) == 5 and parts[:3] == ["grep", "-F", "--"] and parts[3] == check["text"] and parts[4] == path:
            return f"Фактически проверено в Linux-среде: файл {path} содержит требуемый текст"
        if kind in {"python_code_exit_code", "python_exit_code"}:
            if parts and (parts[0] in {"python", "python3", "pytest"} or parts[:2] == ["python", "-m"]):
                if not path or path in parts:
                    return "Терминальная команда завершилась с кодом 0 без тайм-аута"
        if kind == "python_tests_passed" and parts and (parts[0] == "pytest" or parts[:3] == ["python", "-m", "pytest"]):
            return "Команда pytest завершилась с кодом 0 без тайм-аута"
        return None

    # Legacy receipts are accepted only to resume tasks/checkpoints created
    # before the terminal-only tool catalog was introduced.
    root = Path(workspace_root).resolve()
    if kind == "python_code_exit_code" and tool_name == "run_python":
        if isinstance(arguments.get("code"), str) and result.get("inline_code") is True and result.get("exit_code") == 0 and result.get("timed_out") is False:
            return "Legacy inline Python call completed with exit_code=0"
    if kind == "directory_exists" and tool_name in {"create_directory", "list_files"}:
        reported = result.get("path", result.get("created"))
        if arguments.get("path") == path or reported == path:
            try:
                if _acceptance_path_candidate(path, root, task_id).is_dir():
                    return f"Фактически проверено: каталог существует — {path}"
            except (OSError, ValueError):
                pass
    if kind in {"file_exists", "file_contains"} and tool_name in {"write_file", "edit_file", "move_file"}:
        affected = arguments.get("destination") if tool_name == "move_file" else arguments.get("path")
        reported = result.get("destination") if tool_name == "move_file" else result.get("path")
        if affected == path and reported == path:
            try:
                candidate = _acceptance_path_candidate(path, root, task_id)
                if candidate.is_file() and (kind == "file_exists" or check["text"] in candidate.read_text(encoding="utf-8")):
                    return f"Legacy filesystem receipt verified: {path}"
            except (OSError, UnicodeError, ValueError):
                pass
    if kind in {"python_exit_code", "python_tests_passed"} and tool_name == "run_python":
        script = result.get("script") or arguments.get("path")
        if script == path and result.get("exit_code") == 0 and result.get("timed_out") is False:
            if kind == "python_exit_code":
                return f"Legacy script completed with exit_code=0: {path}"
            try:
                source = _acceptance_path_candidate(path, root, task_id).read_text(encoding="utf-8")
                if re.search(r"\bassert\b|unittest\.TestCase|pytest\.(raises|mark)|raise\s+AssertionError", source):
                    return f"Legacy test script passed: {path}"
            except (OSError, UnicodeError, ValueError):
                pass
    return None


def verify_acceptance_check_state(
    check: Any,
    workspace_root: str | Path,
    messages: list[dict[str, Any]] | None = None,
    task_id: str | None = None,
) -> str | None:
    """Verify only from recorded terminal receipts; sandbox files are not host files."""
    if not _valid_acceptance_check(check):
        return None
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "tool" or message.get("tool_name") != "terminal":
            continue
        internal = message.get("_localagent_internal")
        if isinstance(internal, dict):
            arguments = {"command": internal.get("_terminal_command") or internal.get("command")}
            proof = verify_acceptance_check(check, "terminal", arguments, internal, workspace_root, task_id)
            if proof:
                return proof
            continue
        try:
            wrapper = json.loads(message.get("content", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(wrapper, dict) or wrapper.get("ok") is not True:
            continue
        result = wrapper.get("result")
        arguments = {"command": result.get("command")} if isinstance(result, dict) else {}
        proof = verify_acceptance_check(check, "terminal", arguments, result, workspace_root, task_id)
        if proof:
            return proof
    # Preserve old checkpoint receipts without treating host workspace files as
    # evidence for new conversation containers.
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        tool_name = message.get("tool_name")
        if tool_name not in {"write_file", "edit_file", "move_file", "create_directory", "list_files", "run_python"}:
            continue
        try:
            wrapper = json.loads(message.get("content", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(wrapper, dict) or wrapper.get("ok") is not True:
            continue
        args = message.get("tool_arguments", {})
        result = wrapper.get("result")
        if not isinstance(result, dict):
            continue
        if not isinstance(args, dict):
            args = {}
        if tool_name == "run_python" and result.get("script"):
            args.setdefault("path", result.get("script"))
        if tool_name in {"create_directory", "list_files"} and result.get("path"):
            args.setdefault("path", result.get("path"))
        if tool_name in {"write_file", "edit_file"} and result.get("path"):
            args.setdefault("path", result.get("path"))
        proof = verify_acceptance_check(check, tool_name, args, result, workspace_root, task_id)
        if proof:
            return proof
    if task_id is None or str(check.get("path", "")).startswith("/work/"):
        # Direct legacy API compatibility. New container paths are verified by
        # terminal receipts; old /work task paths are only checked in their
        # unique historical host task directory.
        root = Path(workspace_root).resolve()
        try:
            candidate = _acceptance_path_candidate(check["path"], root, task_id)
            kind = check["type"]
            if kind == "directory_exists" and candidate.is_dir():
                return f"Фактически проверено: каталог существует — {check['path']}"
            if kind in {"file_exists", "file_contains"} and candidate.is_file():
                if kind == "file_exists" or check["text"] in candidate.read_text(encoding="utf-8"):
                    return f"Фактически проверено: файл существует — {check['path']}"
        except (OSError, UnicodeError, ValueError):
            pass
    return None

def create_plan(goal: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Create and validate a new structured plan."""
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("План должен содержать цель")
    if not isinstance(steps, list) or not steps:
        raise ValueError("План должен содержать хотя бы один этап")

    normalized = []
    seen_ids = set()

    for index, item in enumerate(steps, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Этап {index}: ожидался объект")

        step_id = item.get("id", f"step_{index}")
        description = item.get("description")
        expected_result = item.get("expected_result")

        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError(f"Этап {index}: некорректный id")
        if step_id in seen_ids:
            raise ValueError(f"Повторяющийся id этапа: {step_id}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Этап {index}: отсутствует описание")
        if not isinstance(expected_result, str) or not expected_result.strip():
            raise ValueError(f"Этап {index}: отсутствует ожидаемый результат")

        check = item.get("acceptance_check")
        if check is not None and not _valid_acceptance_check(check):
            raise ValueError(
                f"Этап {index}: некорректный acceptance_check"
            )

        success_criteria = item.get("success_criteria")
        if success_criteria is None:
            success_criteria = [expected_result.strip()]
        if (
            not isinstance(success_criteria, list)
            or not success_criteria
            or any(
                not isinstance(criterion, str) or not criterion.strip()
                for criterion in success_criteria
            )
        ):
            raise ValueError(
                f"Этап {index}: success_criteria должен быть непустым списком строк"
            )

        task_prompt = item.get("task_prompt")
        if task_prompt is not None and (
            not isinstance(task_prompt, str) or not task_prompt.strip()
        ):
            raise ValueError(
                f"Этап {index}: task_prompt должен быть непустой строкой или null"
            )

        dependencies = item.get("dependencies", [])
        if (
            not isinstance(dependencies, list)
            or any(not isinstance(dep, str) or not dep.strip() for dep in dependencies)
            or len(dependencies) != len(set(dependencies))
        ):
            raise ValueError(
                f"Этап {index}: dependencies должен быть списком уникальных идентификаторов"
            )
        unavailable_dependencies = [dep for dep in dependencies if dep not in seen_ids]
        if unavailable_dependencies:
            raise ValueError(
                f"Этап {index}: зависимости должны ссылаться на предыдущие этапы: "
                f"{', '.join(unavailable_dependencies)}"
            )

        seen_ids.add(step_id)
        normalized.append({
            "id": step_id,
            "description": description.strip(),
            "expected_result": expected_result.strip(),
            "success_criteria": [criterion.strip() for criterion in success_criteria],
            "task_prompt": task_prompt.strip() if isinstance(task_prompt, str) else None,
            "dependencies": list(dependencies),
            "acceptance_check": check,
            "status": "pending",
            "verification": None,
        })

    normalized[0]["status"] = "in_progress"

    return {
        "version": PLAN_VERSION,
        "goal": goal.strip(),
        "status": "in_progress",
        "current_step_id": normalized[0]["id"],
        "steps": normalized,
    }


def validate_plan(plan: Any) -> bool:
    """Return whether a stored plan has a valid structure."""
    if not isinstance(plan, dict) or plan.get("version") != PLAN_VERSION:
        return False
    if not isinstance(plan.get("goal"), str) or not plan["goal"].strip():
        return False
    if plan.get("status") not in PLAN_STATUSES:
        return False

    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return False

    seen_ids = set()
    for step in steps:
        if not isinstance(step, dict):
            return False
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id or step_id in seen_ids:
            return False
        dependencies = step.get("dependencies", [])
        if (
            not isinstance(dependencies, list)
            or any(not isinstance(dep, str) or dep not in seen_ids for dep in dependencies)
            or len(dependencies) != len(set(dependencies))
        ):
            return False
        seen_ids.add(step_id)
        if not isinstance(step.get("description"), str) or not step["description"].strip():
            return False
        if not isinstance(step.get("expected_result"), str) or not step["expected_result"].strip():
            return False
        if step.get("status") not in STEP_STATUSES:
            return False
        success_criteria = step.get("success_criteria")
        if success_criteria is not None and (
            not isinstance(success_criteria, list)
            or not success_criteria
            or any(
                not isinstance(criterion, str) or not criterion.strip()
                for criterion in success_criteria
            )
        ):
            return False
        task_prompt = step.get("task_prompt")
        if task_prompt is not None and (
            not isinstance(task_prompt, str) or not task_prompt.strip()
        ):
            return False
        dependencies = step.get("dependencies")
        if dependencies is not None and (
            not isinstance(dependencies, list)
            or any(not isinstance(dep, str) or not dep.strip() for dep in dependencies)
        ):
            return False
        verification = step.get("verification")
        if verification is not None and not isinstance(verification, str):
            return False
        check = step.get("acceptance_check")
        if check is not None and not _valid_acceptance_check(check):
            return False
        completion_mode = step.get("completion_mode")
        if completion_mode is not None and completion_mode not in {"tool", "response", "verified"}:
            return False
        if check is not None and completion_mode not in (None, "verified"):
            return False

    current = plan.get("current_step_id")
    if current is not None and current not in seen_ids:
        return False
    return True



def reconcile_plan_with_workspace(
    plan: dict[str, Any],
    workspace_root: str | Path,
    messages: list[dict[str, Any]] | None = None,
    task_id: str | None = None,
    after_step_id: str | None = None,
) -> list[dict[str, str]]:
    """Mark later stages complete only when their own deterministic checks pass.

    This is deliberately conservative: stages without an explicit machine-checkable
    acceptance_check are never inferred complete from filenames or model claims.
    """
    if not validate_plan(plan):
        raise ValueError("Некорректный план")
    reconciled: list[dict[str, str]] = []
    anchor_id = after_step_id or (current_step(plan) or {}).get("id")
    if anchor_id is None:
        return reconciled
    current_index = next(
        (i for i, item in enumerate(plan["steps"]) if item["id"] == anchor_id),
        None,
    )
    if current_index is None:
        raise ValueError(f"Этап для согласования не найден: {anchor_id}")
    completed_ids = {item["id"] for item in plan["steps"] if item["status"] in {"completed", "skipped"}}
    for item in plan["steps"][current_index + 1:]:
        if item["status"] == "completed" or not item.get("acceptance_check"):
            continue
        if any(dep not in completed_ids for dep in item.get("dependencies", [])):
            continue
        proof = verify_acceptance_check_state(
            item["acceptance_check"], workspace_root, messages, task_id
        )
        if not proof:
            continue
        item["status"] = "completed"
        item["verification"] = "Reconciled from recorded terminal evidence: " + proof
        completed_ids.add(item["id"])
        reconciled.append({"step_id": item["id"], "verification": item["verification"]})
    unfinished = next((item for item in plan["steps"] if item["status"] not in {"completed", "skipped"}), None)
    plan["current_step_id"] = unfinished["id"] if unfinished else None
    plan["status"] = "in_progress" if unfinished else "completed"
    if unfinished is not None and unfinished["status"] == "pending":
        unfinished["status"] = "in_progress"
    return reconciled

def reconcile_plan_with_review_evidence(
    plan: dict[str, Any],
    reviewed_steps: list[dict[str, Any]],
    tool_evidence: list[dict[str, Any]],
    workspace_root: str | Path,
    messages: list[dict[str, Any]] | None = None,
    task_id: str | None = None,
    after_step_id: str | None = None,
) -> list[dict[str, str]]:
    """Reconcile scope-leaked work identified by the step reviewer.

    The reviewer may identify later stages already completed while working on the
    current stage. Only existing future step IDs are considered. Successful tool
    receipts are mandatory; stages with deterministic acceptance checks additionally
    require those checks to pass locally. Dependencies are evaluated in plan order.
    """
    if not validate_plan(plan):
        raise ValueError("Некорректный план")
    if not isinstance(reviewed_steps, list) or not reviewed_steps or not tool_evidence:
        return []

    anchor_id = after_step_id or (current_step(plan) or {}).get("id")
    current_index = next(
        (i for i, item in enumerate(plan["steps"]) if item["id"] == anchor_id),
        None,
    )
    if current_index is None:
        return []

    requested: dict[str, str] = {}
    for item in reviewed_steps:
        if not isinstance(item, dict):
            continue
        step_id = item.get("step_id")
        verification = item.get("verification")
        if isinstance(step_id, str) and step_id and isinstance(verification, str) and verification.strip():
            requested[step_id] = verification.strip()[:1200]

    completed_ids = {
        item["id"] for item in plan["steps"] if item.get("status") in {"completed", "skipped"}
    }
    reconciled: list[dict[str, str]] = []
    for item in plan["steps"][current_index + 1:]:
        step_id = item.get("id")
        verification = requested.get(step_id)
        if not verification or item.get("status") in {"completed", "skipped", "question"}:
            continue
        if any(dep not in completed_ids for dep in item.get("dependencies", [])):
            continue

        check = item.get("acceptance_check")
        if check is not None:
            proof = verify_acceptance_check_state(check, workspace_root, messages, task_id)
            if not proof:
                continue
            verification = proof

        item["status"] = "completed"
        item["verification"] = "Reconciled by step reviewer against successful tool evidence: " + verification
        completed_ids.add(step_id)
        reconciled.append({"step_id": step_id, "verification": item["verification"]})

    unfinished = next((item for item in plan["steps"] if item["status"] not in {"completed", "skipped"}), None)
    plan["current_step_id"] = unfinished["id"] if unfinished else None
    plan["status"] = "in_progress" if unfinished else "completed"
    if unfinished is not None and unfinished.get("status") == "pending":
        unfinished["status"] = "in_progress"
    return reconciled


def current_step(plan: dict[str, Any]) -> dict[str, Any] | None:
    """Return the current unfinished step."""
    if not validate_plan(plan):
        raise ValueError("Некорректный план")

    for step in plan["steps"]:
        if step["status"] not in {"completed", "skipped"}:
            return step
    return None


def set_step_status(
    plan: dict[str, Any],
    step_id: str,
    status: str,
    verification: str | None = None,
) -> dict[str, Any]:
    """Update one step and recalculate plan progress."""
    if not validate_plan(plan):
        raise ValueError("Некорректный план")
    if status not in STEP_STATUSES:
        raise ValueError(f"Недопустимый статус этапа: {status}")

    target = next((s for s in plan["steps"] if s["id"] == step_id), None)
    if target is None:
        raise ValueError(f"Этап не найден: {step_id}")
    if status == "completed" and (
        not isinstance(verification, str) or not verification.strip()
    ):
        raise ValueError("Для завершения этапа требуется подтверждение результата")

    target["status"] = status
    target["verification"] = verification.strip() if verification else None

    unfinished = next(
        (s for s in plan["steps"] if s["status"] not in {"completed", "skipped"}),
        None,
    )
    plan["current_step_id"] = unfinished["id"] if unfinished else None

    if unfinished is None:
        plan["status"] = "completed"
    elif status == "question":
        plan["status"] = "paused"
    else:
        plan["status"] = "in_progress"

    if unfinished is not None and unfinished["status"] == "pending":
        unfinished["status"] = "in_progress"

    return plan


def pause_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Mark a valid, incomplete plan as paused."""
    if not validate_plan(plan):
        raise ValueError("Некорректный план")
    if any(s["status"] not in {"completed", "skipped"} for s in plan["steps"]):
        plan["status"] = "paused"
    return plan
