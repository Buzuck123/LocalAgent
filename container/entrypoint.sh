#!/bin/bash
set -Eeuo pipefail

: "${SWEREX_AUTH_TOKEN:?SWEREX_AUTH_TOKEN is required}"

# Verify the declared command, Python and browser capabilities for this container.
# Optional capability gaps are recorded but do not prevent the sandbox from starting.
python3 /usr/local/bin/localagent-check-capabilities.py > /etc/localagent-capabilities-startup.log

# The Python supervisor scrubs the initial environment and protects the
# SWE-ReX server process before it starts any interactive Bash sessions.
exec python3 /usr/local/bin/localagent-swerex-server.py
