"""Shared orchestration primitives for tool contracts, recovery and task state.

This module is deliberately deterministic: the model proposes actions, while the
host validates contracts, classifies failures and records durable observations.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any


class ToolContractError(ValueError):
    """A model-proposed tool call violates the registered tool contract."""


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True  # Unknown JSON Schema types are left to the tool implementation.


def _validate_schema(value: Any, schema: dict, path: str = "arguments") -> None:
    if not isinstance(schema, dict):
        return
    if isinstance(schema.get("oneOf"), list):
        valid = 0
        for candidate in schema["oneOf"]:
            try:
                _validate_schema(value, candidate, path)
                valid += 1
            except ToolContractError:
                pass
        if valid != 1:
            raise ToolContractError(f"{path}: value must match exactly one oneOf schema")
    if isinstance(schema.get("not"), dict):
        try:
            _validate_schema(value, schema["not"], path)
        except ToolContractError:
            pass
        else:
            raise ToolContractError(f"{path}: value matches forbidden schema")
    expected = schema.get("type")
    if isinstance(expected, str) and not _type_matches(value, expected):
        raise ToolContractError(f"{path}: expected {expected}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolContractError(f"{path}: value must be one of {schema['enum']!r}")
    if isinstance(value, str):
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            raise ToolContractError(f"{path}: string is shorter than minLength")
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            raise ToolContractError(f"{path}: string exceeds maxLength")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            raise ToolContractError(f"{path}: value is below minimum {schema['minimum']}")
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            raise ToolContractError(f"{path}: value exceeds maximum {schema['maximum']}")
    if isinstance(value, dict):
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        missing = [key for key in required if key not in value]
        if missing:
            raise ToolContractError(f"{path}: missing required field(s): {', '.join(map(str, missing))}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ToolContractError(f"{path}: unexpected field(s): {', '.join(extra)}")
        for key, item in value.items():
            if key in properties:
                _validate_schema(item, properties[key], f"{path}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate_schema(item, schema["items"], f"{path}[{index}]")


def validate_tool_call(name: Any, arguments: Any, tool_definitions: list[dict]) -> dict:
    """Validate name and arguments against the same schemas exposed to the LLM."""
    if not isinstance(name, str) or not name:
        raise ToolContractError("Tool name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise ToolContractError("Tool arguments must be a JSON object")
    definition = next((item.get("function", {}) for item in tool_definitions
                       if isinstance(item, dict)
                       and isinstance(item.get("function"), dict)
                       and item["function"].get("name") == name), None)
    if definition is None:
        raise ToolContractError(f"Unknown tool: {name}")
    schema = definition.get("parameters", {"type": "object"})
    _validate_schema(arguments, schema)
    return {"name": name, "arguments": arguments, "schema_validated": True}


# These names describe effects, not permissions. The tool's own sandbox remains
# the security boundary; this classification controls recovery and repeat policy.
READ_ONLY_TOOLS = {"web_search", "open_browser_page", "get_tool_result", "load_file_for_model"}
MUTATING_TOOLS = {"terminal", "publish_file"}


def classify_failure(tool_name: str, arguments: dict, error_text: str) -> dict:
    """Return a stable, machine-readable error category and recovery policy."""
    text = str(error_text or "")
    low = text.lower()
    structured = {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            structured = parsed
            if isinstance(parsed.get("result"), dict):
                structured = {**parsed["result"], **parsed}
    except (TypeError, json.JSONDecodeError):
        pass
    timed_out = structured.get("timed_out") is True or bool(
        re.search(r"\btimed_out[\"']?\s*[:=]\s*true\b", low)
    )
    exit_code = structured.get("exit_code")
    nonzero_exit = isinstance(exit_code, (int, float)) and not isinstance(exit_code, bool) and exit_code != 0
    if any(x in low for x in ("missing required field", "unexpected field", "expected string", "expected integer", "unknown tool", "must be a json object")):
        category, action = "invalid_tool_call", "correct_arguments"
    elif any(x in low for x in ("fileexistserror", "already exists", "file exists", "duplicate column", "already present")):
        category, action = "state_conflict", "inspect_state_before_retry"
    elif any(x in low for x in ("filenotfounderror", "no such file", "not found", "no such table", "no such column")):
        category, action = "missing_resource", "discover_actual_resources"
    elif any(x in low for x in ("syntax error", "invalid sql", "malformed", "parse error")):
        category, action = "invalid_expression", "revise_expression_after_inspection"
    elif any(x in low for x in ("permissionerror", "permission denied", "read-only", "readonly", "path escapes", "outside workspace")):
        category, action = "access_boundary", "use_permitted_path_or_replan"
    elif timed_out or "timeout" in low or "timed out" in low:
        category = "timeout"
        recovered = structured.get("shell_recovered") is True
        action = "inspect_state_after_timeout" if tool_name == "terminal" and recovered else (
            "restore_terminal_before_retry" if tool_name == "terminal" else "reduce_scope_or_retry_once"
        )
    elif any(x in low for x in ("modulenotfounderror", "importerror", "no module named")):
        category, action = "environment_dependency", "check_available_runtime"
    elif nonzero_exit or any(x in low for x in ("assertionerror", "assertion failed", "verification failed", "exit_code=1", '"exit_code": 1')):
        category, action = "verification_failure", "inspect_evidence_and_fix_cause"
    elif any(x in low for x in ("connectionerror", "connection refused", "temporarily unavailable", "503", "429")):
        category, action = "transient_service", "bounded_retry_if_safe"
    else:
        category, action = "tool_failure", "diagnose_then_change_strategy"

    effect = "read_only" if tool_name in READ_ONLY_TOOLS else ("mutating" if tool_name in MUTATING_TOOLS else "unknown")
    # A transient error does not make a write safe to repeat: the remote/local
    # operation may have committed before the acknowledgement was lost.
    retry_safe = effect == "read_only"
    if action == "inspect_state_before_retry":
        retry_safe = False
    return {
        "category": category,
        "recovery_action": action,
        "tool_effect": effect,
        "retry_safe_without_state_check": retry_safe,
        "same_call_should_repeat": False,
        "error_excerpt": re.sub(r"\s+", " ", text).strip()[:1000],
    }



def migrate_simple_action_evidence_to_plan(plan: dict[str, Any], step_tool_evidence: dict[str, Any]) -> int:
    """Carry successful pre-plan tool receipts into the first planned stage.

    Receipts from ``simple_action`` are task-level observations, not proof that a
    particular stage is complete. The explicit marker lets the reviewer distinguish
    them from evidence generated while executing the new stage. The operation is
    idempotent so a restored checkpoint cannot duplicate migrated receipts.
    """
    if not isinstance(plan, dict) or not isinstance(step_tool_evidence, dict):
        return 0
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return 0
    target = next(
        (step for step in steps if isinstance(step, dict) and step.get("status") != "completed"),
        None,
    )
    source = step_tool_evidence.get("simple_action")
    if not target or target.get("id") == "simple_action" or not isinstance(source, list) or not source:
        return 0

    target_id = target.get("id")
    if not isinstance(target_id, str) or not target_id:
        return 0
    destination = step_tool_evidence.setdefault(target_id, [])
    if not isinstance(destination, list):
        destination = []
        step_tool_evidence[target_id] = destination

    existing = {
        json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        for item in destination if isinstance(item, dict)
        and item.get("migrated_from_step") == "simple_action"
    }
    migrated = 0
    for receipt in source:
        if not isinstance(receipt, dict):
            continue
        item = dict(receipt)
        item["migrated_from_step"] = "simple_action"
        item["evidence_scope"] = "task_level_before_planning"
        fingerprint = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        if fingerprint in existing:
            continue
        destination.append(item)
        existing.add(fingerprint)
        migrated += 1
    step_tool_evidence[target_id] = destination[-20:]
    return migrated


def tool_call_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    """Stable fingerprint for an exact tool call, independent of dict key order."""
    try:
        payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        payload = repr(arguments)
    return f"{tool_name}|{payload}"


def _json_arguments(value: Any) -> dict[str, Any]:
    """Read argument snapshots from both current and legacy checkpoint entries."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, json.JSONDecodeError):
            pass
    return {}


def _canonical_recovery_path(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value == "work":
        return "/work"
    if value.startswith("work/"):
        return "/work/" + value[len("work/"):]
    if value == "/workspace":
        return "."
    if value.startswith("/workspace/"):
        return value[len("/workspace/"):]
    return value


def _is_missing_resource_error(error: str) -> bool:
    """Recognize common Python/OS missing-path diagnostics."""
    low = str(error or "").lower()
    return any(token in low for token in (
        "filenotfounderror", "enoent", "no such file", "file not found",
        "directory not found", "not found",
    ))


def _successful_action_repairs_failure(failure: dict, success: dict) -> bool:
    """Return true only when a later mutation fixes the failed call's missing path."""
    failed_tool = str(failure.get("tool_name", ""))
    success_tool = str(success.get("tool_name", ""))
    failed_args = _json_arguments(failure.get("arguments"))
    success_args = _json_arguments(success.get("arguments"))
    error = str(failure.get("error", "")).lower()

    if failed_tool == "terminal" and success_tool == "terminal":
        failed_command = str(failed_args.get("command", ""))
        success_command = str(success_args.get("command", ""))
        if not failed_command or not success_command or failed_command == success_command:
            return False
        if "no module named" in error or "modulenotfounderror" in error:
            module = re.search(r"no module named ['\"]?([a-zA-Z0-9_.-]+)", error)
            return bool(module and re.search(r"\b(pip|pip3|apt|apt-get)\s+install\b", success_command)
                        and module.group(1).lower() in success_command.lower())
        if _is_missing_resource_error(error):
            path_tokens = re.findall(r"(?:/[^\s'\"]+|[\w.-]+(?:/[\w.-]+)+)", failed_command)
            mutating = re.search(r"\b(mkdir|touch|cp|mv|tee|install|chmod|chown)\b|(?:>|>>)", success_command)
            return bool(mutating and any(path in success_command for path in path_tokens))
        if "permission denied" in error or "permissionerror" in error:
            path_tokens = re.findall(r"(?:/[^\s'\"]+|[\w.-]+(?:/[\w.-]+)+)", failed_command)
            return bool(re.search(r"\b(chmod|chown|sudo)\b", success_command)
                        and any(path in success_command for path in path_tokens))
        return False

    # A terminal action can create or repair a file that a later model-inspection
    # call needs. Only clear the failure when the command visibly targets that path.
    if failed_tool == "load_file_for_model" and success_tool == "terminal":
        if not _is_missing_resource_error(error):
            return False
        failed_path = _canonical_recovery_path(failed_args.get("path"))
        command = str(success_args.get("command", ""))
        mutating = re.search(r"\b(mkdir|touch|cp|mv|tee|install|chmod|chown|curl|wget)\b|(?:>|>>)", command)
        return bool(failed_path and mutating and failed_path in command)

    return False


def should_block_repeated_call(tool_name: str, arguments: dict[str, Any], history: Any) -> dict[str, Any] | None:
    """Block an exact repeat unless a later successful action repaired its cause.

    An unrelated success never clears a failure. A narrowly matched terminal
    state change can clear a missing-file error before model inspection is retried.
    """
    if not isinstance(history, list):
        return None
    fingerprint = tool_call_fingerprint(tool_name, arguments)
    matching_indexes = [index for index, item in enumerate(history)
                        if isinstance(item, dict) and item.get("call_fingerprint") == fingerprint
                        and item.get("status") != "success"]
    if not matching_indexes:
        return None
    last_failure_index = matching_indexes[-1]
    previous = history[last_failure_index]
    later_successes = [item for item in history[last_failure_index + 1:]
                       if isinstance(item, dict) and item.get("status") == "success"]
    if any(item.get("call_fingerprint") == fingerprint for item in later_successes):
        return None
    if any(_successful_action_repairs_failure(previous, item) for item in later_successes):
        return None
    return {
        "blocked": True,
        "category": "repeated_failed_call",
        "recovery_action": "inspect_state_or_change_strategy",
        "retry_safe_without_state_check": False,
        "same_call_should_repeat": False,
        "repeat_count": len(matching_indexes) + 1,
        "previous_error": str(previous.get("error", ""))[:500],
        "message": (
            "Этот точный вызов инструмента уже завершился ошибкой. Повтор заблокирован: "
            "сначала проверь состояние/результат предыдущей операции или измени аргументы и стратегию."
        ),
    }

def append_execution_event(checkpoint: dict, event_type: str, *, tool_name: str | None = None,
                           detail: str = "", status: str | None = None) -> dict:
    """Append a bounded, timestamped event to the durable task checkpoint."""
    if not isinstance(checkpoint, dict):
        return {}
    events = checkpoint.get("execution_events")
    if not isinstance(events, list):
        events = []
    event = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "type": str(event_type)[:60],
        "tool": str(tool_name)[:100] if tool_name else None,
        "status": str(status)[:40] if status else None,
        "detail": re.sub(r"\s+", " ", str(detail or "")).strip()[:500],
    }
    events.append(event)
    checkpoint["execution_events"] = events[-100:]
    return event


def update_task_state(checkpoint: dict, *, tool_name: str | None = None,
                      arguments: dict | None = None, result: Any = None,
                      success: bool | None = None, failure: dict | None = None) -> dict:
    """Maintain a compact current-state summary separately from the event history."""
    state = checkpoint.get("task_state")
    if not isinstance(state, dict):
        state = {"actions_total": 0, "actions_succeeded": 0, "actions_failed": 0,
                 "successful_tools": [], "failed_categories": {}, "unresolved_errors": []}
    if tool_name:
        state["actions_total"] = int(state.get("actions_total", 0) or 0) + 1
        state["last_tool"] = tool_name
        state["last_arguments_keys"] = sorted(list(arguments or {}))[:30]
        state["last_action_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if success is True:
            state["actions_succeeded"] = int(state.get("actions_succeeded", 0) or 0) + 1
            names = state.get("successful_tools")
            if not isinstance(names, list):
                names = []
            if tool_name not in names:
                names.append(tool_name)
            state["successful_tools"] = names[-50:]
            # A successful follow-up using the same tool is evidence that its
            # previous tool-level failure was recovered; retain other failures.
            unresolved = state.get("unresolved_errors")
            if isinstance(unresolved, list):
                success_receipt = {"tool_name": tool_name, "arguments": arguments or {}}
                remaining = []
                for item in unresolved:
                    if not isinstance(item, dict):
                        remaining.append(item)
                        continue
                    # Preserve the existing same-tool recovery behavior, but also
                    # clear a missing-resource error when a later successful write
                    # creates the exact path that the failed read/run needed.
                    if item.get("tool") == tool_name:
                        continue
                    failure_receipt = {
                        "tool_name": item.get("tool"),
                        "arguments": item.get("arguments", {}),
                        "error": item.get("error", ""),
                    }
                    if _successful_action_repairs_failure(failure_receipt, success_receipt):
                        continue
                    # A verified copy to workspace can recover a previous access
                    # boundary error from artifact publication.
                    if tool_name == "copy_file" and item.get("category") == "access_boundary":
                        continue
                    remaining.append(item)
                state["unresolved_errors"] = remaining[-20:]
            state["last_error"] = None
        elif success is False:
            state["actions_failed"] = int(state.get("actions_failed", 0) or 0) + 1
            category = str((failure or {}).get("category") or "tool_failure")
            categories = state.get("failed_categories")
            if not isinstance(categories, dict):
                categories = {}
            categories[category] = int(categories.get(category, 0) or 0) + 1
            state["failed_categories"] = categories
            recovery_args = {
                key: (arguments or {})[key]
                for key in ("path", "source", "destination", "destination_path")
                if isinstance(arguments, dict) and key in arguments
                and isinstance(arguments[key], (str, int, float, bool))
            }
            error_item = {"tool": tool_name, "category": category,
                          "error": str((failure or {}).get("error_excerpt") or "")[:500],
                          "recovery_action": (failure or {}).get("recovery_action"),
                          "arguments": recovery_args}
            unresolved = state.get("unresolved_errors")
            if not isinstance(unresolved, list):
                unresolved = []
            unresolved.append(error_item)
            state["unresolved_errors"] = unresolved[-20:]
            state["last_error"] = error_item
    checkpoint["task_state"] = state
    return state
