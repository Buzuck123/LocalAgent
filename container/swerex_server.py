#!/usr/bin/env python3
"""Start SWE-ReX without exposing its API token to interactive Bash sessions."""
from __future__ import annotations

import ctypes
import os
import pwd
import socket
import subprocess
import time


def take_and_scrub_auth_token() -> str:
    token = os.environ.get("SWEREX_AUTH_TOKEN")
    if not token:
        raise RuntimeError("SWEREX_AUTH_TOKEN is required")

    # Save the original environment string address before unsetenv compacts the
    # environment pointer array. Zero its bytes as well: /proc/<pid>/environ
    # reads the original execve memory area on Linux.
    libc = ctypes.CDLL(None)
    envp = ctypes.POINTER(ctypes.c_void_p).in_dll(libc, "environ")
    token_ptr = None
    token_len = 0
    index = 0
    while envp[index]:
        pointer = envp[index]
        entry = ctypes.string_at(pointer)
        if entry.startswith(b"SWEREX_AUTH_TOKEN="):
            token_ptr = pointer
            token_len = len(entry)
            break
        index += 1

    os.environ.pop("SWEREX_AUTH_TOKEN", None)
    if token_ptr:
        ctypes.memset(token_ptr, 0, token_len)

    # Prevent same-UID processes inside the container from reading this
    # server's memory through procfs/ptrace. The token remains only in the
    # server's Python object state and is never inherited by child processes.
    if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE = 4
        raise RuntimeError("Could not protect SWE-ReX server process memory")
    return token


def main() -> None:
    token = take_and_scrub_auth_token()
    import swerex.server as swerex_server
    swerex_server.AUTH_TOKEN = token
    del token

    log_dir = "/var/log/privoxy"
    os.makedirs(log_dir, exist_ok=True)
    privoxy_user = pwd.getpwnam("privoxy")
    os.chown(log_dir, privoxy_user.pw_uid, privoxy_user.pw_gid)
    privoxy = subprocess.Popen(
        ["/usr/sbin/privoxy", "--no-daemon", "/etc/privoxy/config"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    try:
        ready = False
        for _ in range(50):
            if privoxy.poll() is not None:
                raise RuntimeError("Privoxy failed to start")
            try:
                with socket.create_connection(("127.0.0.1", 8118), timeout=0.2):
                    ready = True
                    break
            except OSError:
                time.sleep(0.1)
        if not ready:
            raise RuntimeError("Privoxy did not become ready on 127.0.0.1:8118")

        import uvicorn
        uvicorn.run("swerex.server:app", host="0.0.0.0", port=8000, log_level="info")
    finally:
        if privoxy.poll() is None:
            privoxy.terminate()
            try:
                privoxy.wait(timeout=5)
            except subprocess.TimeoutExpired:
                privoxy.kill()
                privoxy.wait(timeout=5)


if __name__ == "__main__":
    main()
