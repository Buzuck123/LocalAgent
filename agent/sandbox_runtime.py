"""Execution abstraction and SWE-ReX 1.4.0 implementation.

This module deliberately knows nothing about Podman or conversation storage.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SandboxRuntime(Protocol):
    def is_alive(self) -> bool: ...
    def ensure_session(self, session: str = "main") -> bool: ...
    def run(self, command: str, *, session: str = "main", timeout: int = 120) -> dict[str, Any]: ...
    def start_operation(self, command: str, *, session: str = "main", timeout: int | None = None,
                        idempotency_key: str | None = None) -> dict[str, Any]: ...
    def get_operation(self, operation_id: str, *, session: str = "main", cursor: int = 0,
                      max_output_chars: int = 65536, wait_timeout: float = 0.0) -> dict[str, Any]: ...
    def interrupt_operation(self, operation_id: str, *, session: str = "main") -> dict[str, Any]: ...
    def send_input(self, operation_id: str, value: str, *, session: str = "main", sendline: bool = True) -> dict[str, Any]: ...
    def resize_terminal(self, *, session: str = "main", rows: int = 24, cols: int = 80) -> None: ...
    def close_session(self, session: str = "main") -> None: ...
    def upload_file(self, source_path: str, target_path: str) -> None: ...


class SwerexRemoteRuntime:
    """Synchronous adapter over SWE-ReX's asynchronous RemoteRuntime API."""

    def __init__(self, *, host: str, port: int, auth_token: str, loop_runner, timeout: float = 10.0):
        from swerex.runtime.remote import RemoteRuntime
        self._runner = loop_runner
        self._runtime = RemoteRuntime(
            auth_token=auth_token, host=host, port=port, timeout=timeout,
        )

    def _call(self, awaitable, timeout: float = 180):
        return self._runner(awaitable, timeout=timeout)

    @staticmethod
    def _name(exc: BaseException) -> str:
        return type(exc).__name__

    @staticmethod
    def _is_timeout_exception(exc: BaseException) -> bool:
        name = type(exc).__name__.lower()
        return isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in str(exc).lower()

    def is_alive(self) -> bool:
        try:
            response = self._call(self._runtime.is_alive(timeout=2.0), timeout=5)
            return bool(getattr(response, "is_alive", response))
        except Exception:
            return False

    def ensure_session(self, session: str = "main") -> bool:
        """Ensure a Bash session exists; return True only when a new session was created."""
        from swerex.runtime.local import CreateBashSessionRequest
        try:
            self._call(self._runtime.create_session(CreateBashSessionRequest(session=session)), timeout=30)
            return True
        except Exception as exc:
            if self._name(exc) == "SessionExistsError":
                return False
            raise

    @staticmethod
    def _result_payload(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            data = result
        elif hasattr(result, "model_dump"):
            data = result.model_dump()
        elif hasattr(result, "dict"):
            data = result.dict()
        else:
            data = {
                key: getattr(result, key)
                for key in ("output", "stdout", "stderr", "exit_code", "return_code", "timed_out")
                if hasattr(result, key)
            }
        stdout = data.get("stdout")
        if stdout is None:
            stdout = data.get("output", "")
        stderr = data.get("stderr", "")
        failure_reason = data.get("failure_reason", "")
        if failure_reason and not stderr:
            stderr = failure_reason
        exit_code = data.get("exit_code", data.get("return_code"))
        timed_out = bool(data.get("timed_out", False)) or exit_code is None
        if exit_code is None:
            exit_code = 124 if timed_out else 1
        return {
            "stdout": str(stdout or ""),
            "stderr": str(stderr or ""),
            "exit_code": exit_code if isinstance(exit_code, int) else 1,
            "timed_out": timed_out,
        }

    def _recover_session(self, session: str) -> bool:
        """Replace a potentially desynchronized shell without touching its container."""
        from swerex.runtime.local import BashAction, CreateBashSessionRequest

        try:
            self.close_session(session)
            self._call(
                self._runtime.create_session(CreateBashSessionRequest(session=session)),
                timeout=30,
            )
            probe = self._call(
                self._runtime.run_in_session(
                    BashAction(command="printf '__LOCALAGENT_SHELL_READY__\\n'", session=session,
                               timeout=5, check="silent")
                ),
                timeout=15,
            )
            payload = self._result_payload(probe)
            return payload["exit_code"] == 0 and "__LOCALAGENT_SHELL_READY__" in payload["stdout"]
        except Exception as exc:
            logger.warning("SWE-ReX shell recovery failed (%s)", type(exc).__name__)
            return False

    def run(self, command: str, *, session: str = "main", timeout: int = 120) -> dict[str, Any]:
        from swerex.runtime.local import BashAction

        def invoke():
            try:
                action = BashAction(command=command, session=session, timeout=timeout, check="silent")
            except TypeError:
                action = BashAction(command=command, session=session)
            return self._call(self._runtime.run_in_session(action), timeout=timeout + 60)

        try:
            result = invoke()
        except Exception as exc:
            if self._name(exc) == "SessionDoesNotExistError":
                self.ensure_session(session)
                result = invoke()
            elif self._is_timeout_exception(exc):
                recovered = self._recover_session(session)
                return {
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout} seconds. " +
                              ("The Bash session was reset; filesystem changes already made may remain." if recovered
                               else "The Bash session could not be recovered; further terminal calls are unsafe."),
                    "exit_code": 124,
                    "timed_out": True,
                    "shell_recovered": recovered,
                    "shell_state_reset": recovered,
                }
            else:
                raise

        payload = self._result_payload(result)
        if payload["timed_out"]:
            recovered = self._recover_session(session)
            payload["shell_recovered"] = recovered
            payload["shell_state_reset"] = recovered
            note = ("Bash session reset successfully; filesystem changes already made may remain."
                    if recovered else
                    "Bash session recovery failed; further terminal calls are unsafe.")
            payload["stderr"] = (payload.get("stderr", "") + "\n" + note).strip()
        return payload

    @staticmethod
    def _dump_model(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return dict(vars(value))

    def start_operation(self, command: str, *, session: str = "main", timeout: int | None = None,
                        idempotency_key: str | None = None) -> dict[str, Any]:
        """Start a SWE-ReX background operation without waiting for process exit."""
        from swerex.runtime.abstract import BashAction, StartOperationRequest
        action = BashAction(command=command, session=session, timeout=timeout, check="silent")
        request = StartOperationRequest(action=action, idempotency_key=idempotency_key or uuid.uuid4().hex)
        result = self._call(self._runtime.start_operation(request), timeout=30)
        return self._dump_model(result)

    def get_operation(self, operation_id: str, *, session: str = "main", cursor: int = 0,
                      max_output_chars: int = 65536, wait_timeout: float = 0.0) -> dict[str, Any]:
        from swerex.runtime.abstract import GetOperationRequest
        request = GetOperationRequest(session=session, operation_id=operation_id, cursor=cursor,
                                      max_output_chars=max_output_chars, wait_timeout=wait_timeout)
        result = self._call(self._runtime.get_operation(request), timeout=max(30, wait_timeout + 20))
        return self._dump_model(result)

    def interrupt_operation(self, operation_id: str, *, session: str = "main") -> dict[str, Any]:
        from swerex.runtime.abstract import InterruptOperationRequest
        request = InterruptOperationRequest(session=session, operation_id=operation_id)
        result = self._call(self._runtime.interrupt_operation(request), timeout=15)
        return self._dump_model(result)

    def send_input(self, operation_id: str, value: str, *, session: str = "main", sendline: bool = True) -> dict[str, Any]:
        from swerex.runtime.abstract import SendInputRequest
        request = SendInputRequest(session=session, operation_id=operation_id, input=value, sendline=sendline)
        result = self._call(self._runtime.send_input(request), timeout=15)
        return self._dump_model(result)

    def resize_terminal(self, *, session: str = "main", rows: int = 24, cols: int = 80) -> None:
        from swerex.runtime.abstract import ResizeTerminalRequest
        request = ResizeTerminalRequest(session=session, rows=rows, cols=cols)
        self._call(self._runtime.resize_terminal(request), timeout=15)

    def close_session(self, session: str = "main") -> None:
        from swerex.runtime.local import CloseSessionRequest
        try:
            self._call(self._runtime.close_session(CloseSessionRequest(session=session)), timeout=30)
        except Exception as exc:
            if self._name(exc) != "SessionDoesNotExistError":
                raise

    def upload_file(self, source_path: str, target_path: str) -> None:
        from swerex.runtime.local import UploadRequest
        self._call(self._runtime.upload(UploadRequest(source_path=source_path, target_path=target_path)), timeout=120)

    def detach(self) -> None:
        """Release the local reference; never call RemoteRuntime.close()."""
        self._runtime = None
