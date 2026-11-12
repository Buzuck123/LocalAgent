#!/bin/bash
set -Eeuo pipefail

: "${SWEREX_AUTH_TOKEN:?SWEREX_AUTH_TOKEN is required}"

# The Python supervisor scrubs the initial environment and protects the
# SWE-ReX server process before it starts any interactive Bash sessions.
exec python3 /usr/local/bin/localagent-swerex-server.py
