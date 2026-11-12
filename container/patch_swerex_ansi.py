#!/usr/bin/env python3
from importlib.util import find_spec
from pathlib import Path
import sys

spec = find_spec("swerex")
if spec is None or not spec.submodule_search_locations:
    sys.exit("ERROR: SWE-ReX package not found")

local_py = Path(next(iter(spec.submodule_search_locations))) / "runtime" / "local.py"

if not local_py.is_file():
    sys.exit(f"ERROR: SWE-ReX source not found: {local_py}")

source = local_py.read_text()

old = r'''def _strip_control_chars(s: str) -> str:
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", s).replace("\r\n", "\n")
'''

new = r'''def _strip_control_chars(s: str) -> str:
    # Preserve ANSI escape sequences for full terminal/TUI support.
    return s.replace("\r\n", "\n")
'''

old_count = source.count(old)
new_count = source.count(new)

if old_count == 1:
    local_py.write_text(source.replace(old, new, 1))
    print(f"Patched: {local_py}")
elif new_count == 1:
    print(f"Already patched: {local_py}")
else:
    sys.exit(
        f"ERROR: expected exactly one original or patched function; "
        f"found original={old_count}, patched={new_count}"
    )
