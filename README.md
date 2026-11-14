# LocalAgent

### Version 0.7.2

Версия: 0.7.2

LocalAgent is a local AI agent built around Ollama, a persistent conversation planner, and a conversation-scoped Linux execution environment.

## What's new in 0.7.1

- Bash commands are syntax-checked with `bash -n` before dispatch, preventing unterminated quotes/heredocs from wedging the persistent interactive shell.
- SWE-ReX timeouts trigger a bounded reset and health probe of the conversation's `main` Bash session without recreating the Podman container or deleting files.
- Timeout results report whether shell recovery succeeded; a failed recovery marks the sandbox as unhealthy instead of pretending it is ready.
- Ollama tool-call arguments are normalized strictly and malformed argument payloads become recoverable tool errors instead of crashing the task/checkpoint loop.
- The model sees one terminal tool. LocalAgent waits for command completion, returning the accumulated PTY transcript without appending an exit code; after 10 seconds without new output it returns control to the model, while continuing output can keep one wait open for up to 300 seconds. The model can wait again with an empty terminal call, send interactive input, or interrupt with Ctrl+C. Operation IDs, cursors and background polling remain internal. The command hard timeout is unlimited unless explicitly supplied.
- Cancel closes the active Ollama stream and independently interrupts the remote terminal process; unconfirmed remote cancellation is reported as interrupted, not cancelled.
- The shared Xray proxy is started/reconnected to the configured network and its DNS, SOCKS5 handshake and Privoxy egress are probed from the sandbox before network-dependent terminal commands; recognized mid-command proxy/DNS failures trigger a bounded health recheck but never an automatic command replay.
- The active Ollama model selects Direct vs Plan for new tasks, with a conservative fallback if its response is invalid or unavailable.
- Context maintenance uses stable event IDs, deduplicates only repeated IDs around 50%, forecasts growth around 60%, and compacts around 70% while retaining archived source events.
- Optional Debug Trace and returned-thinking capture can be enabled from the UI; trace events are redacted and exportable per task.
- Context compaction remains a deterministic factual projection with a durable raw archive; LLM rewriting of that summary is not yet enabled because it needs stronger preservation validation.
- `load_file_for_model` is available when the model itself should inspect a file; terminal remains the interface for programmatic processing.

## What's new in 0.6.0

- Replaced the model-facing collection of file/Python/Office/SQLite tools with one persistent Bash terminal.
- Added one rootless Podman container per conversation, created lazily on the first terminal call and reused across messages.
- Added SWE-ReX 1.4.0 `RemoteRuntime`, persistent Bash session `main`, recovery after LocalAgent restarts, and cleanup when a conversation is deleted.
- Added a SQLite migration for sandbox metadata and a dedicated `publish_file` tool for delivering generated files to the user.
- The old Bubblewrap implementation remains in the source tree for rollback/reference, but its specialized tools are no longer exposed to the model and it is not an automatic fallback.

## Architecture

```text
Ollama LLM
   |
   v
LocalAgent agent loop / planner
   |
   v
terminal tool
   |
   v
Sandbox Manager ---- SQLite conversation_sandboxes
   |
   v
SandboxRuntime / SWE-ReX RemoteRuntime
   |
   v
127.0.0.1:<dynamic host port> -> container:8000
   |
   v
rootless Podman container (one per conversation)
   |
   v
persistent Bash session: main
```

The planner does not import Podman or SWE-ReX. It sees the terminal as an ordinary tool and receives the available terminal transcript; the separate exit code remains internal to the controller. Sandbox lifecycle is managed outside the planner.

The container is created on the first `terminal` call for a conversation. It is not created per task or per HTTP request. Its writable container filesystem and Bash shell state survive later messages and LocalAgent restarts. The host's `workspace/work` directory is mounted read-write at `/host-workspace`; this is a deliberate shared workspace boundary, not a full home-directory mount. Do not describe the sandbox as isolated from that host directory.

## Requirements

### Host

- Ubuntu/Linux with Python 3.11+ (the project uses Python 3.12 in its supported development environment).
- Ollama with a working model.
- Rootless Podman configured for the LocalAgent user.
- The existing Podman network `localagent-sandbox-internal-test`.
- The existing image `localhost/localagent-sandbox:ubuntu24.04`.
- The LocalAgent host Python dependencies from `requirements.txt`, including `swe-rex==1.4.0`.

Confirm rootless Podman:

```bash
podman info --format '{{.Host.Security.Rootless}}'
```

Expected output: `true`.

Confirm the pre-created network and image (these commands inspect; they do not change the network):

```bash
podman network inspect localagent-sandbox-internal-test
podman image exists localhost/localagent-sandbox:ubuntu24.04
```

### Container image

The image is based on Ubuntu 24.04 and contains SWE-ReX 1.4.0, Python, Privoxy, Playwright/Chromium, document and spreadsheet libraries, PDF/OCR and image tools, and common Linux utilities. These packages are installed when building the image, not on each conversation startup.

Build or rebuild the image from the `container/` directory context:

```bash
podman build \
  -t localhost/localagent-sandbox:ubuntu24.04 \
  -f container/Containerfile \
  container
```

Rebuilding the image does not recreate existing conversation containers. Existing containers continue to use the image layer from which they were created; new conversations use the rebuilt image.

The current image entrypoint starts Privoxy and the SWE-ReX server. The server launcher removes the token from its inherited environment before creating Bash sessions and marks its own process non-dumpable. The LLM does not receive Podman access, the Podman socket, the auth token, or host-port details.

## Installation

```bash
cd /home/<user>/LocalAgent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/patch_host_swerex.py
cp .env.example .env
```

The host-side patch step is required as well as the patch inside the container image: LocalAgent is the SWE-ReX client, while the container runs the patched SWE-ReX server. The script only patches the pinned `swe-rex==1.4.0` installation, is safe to rerun, and refuses other versions.

Edit `.env` for the Ollama endpoint/model if required. Sandbox defaults match the prepared host infrastructure. Do not put the SWE-ReX auth token in `.env`: Sandbox Manager generates a separate token per conversation and stores it in SQLite.

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| `SANDBOX_ENABLED` | `true` | Enables the new conversation sandbox |
| `PODMAN_PATH` | `podman` | Podman executable |
| `SANDBOX_IMAGE` | `localhost/localagent-sandbox:ubuntu24.04` | Prepared sandbox image |
| `SANDBOX_NETWORK` | `localagent-sandbox-internal-test` | Existing internal Podman network |
| `SANDBOX_CPUS` | `2` | Container CPU limit |
| `SANDBOX_MEMORY` | `4g` | Container memory limit |
| `SANDBOX_CONTAINER_PORT` | `8000` | SWE-ReX port inside the container |
| `SWEREX_TIMEOUT` | `10` | RemoteRuntime connection timeout, seconds |
| `SANDBOX_PROXY_REQUIRED` | `true` | Require the shared Xray container lifecycle check before sandbox creation |
| `XRAY_PROXY_CONTAINER` | `localagent-xray-proxy` | Existing shared proxy container name |
| `DEBUG_LLM_TRACE` | `false` | Enable task-scoped Debug Trace by default |
| `DEBUG_LLM_THINKING` | `false` | Store only thinking text actually returned by Ollama, and only when trace is enabled |

Podman publishes the internal port as `127.0.0.1::8000`; Podman selects the host port. LocalAgent reads the actual port from `podman inspect`. The token is passed as `SWEREX_AUTH_TOKEN` to the container at creation time.

## Running

Web interface:

```bash
source .venv/bin/activate
python -m agent.web
```

CLI:

```bash
source .venv/bin/activate
python -m agent.cli
```

The web service binds to `127.0.0.1` by default. Do not expose it to a LAN or the public internet without adding authentication and reviewing the threat model.

## Agent computer tools

The model-facing tool catalog is intentionally small:

- `terminal` — the only computer interface. It runs Bash commands in the persistent Linux container and supports waiting, interactive input, and interruption through the same tool. Operation IDs and terminal lifecycle details remain internal.
- `load_file_for_model` — direct model inspection of supported files at readable container paths. Text documents are delivered in sequential paragraph-aware chunks (`start`/`length`, `next_start`); use terminal and installed libraries for programmatic processing. Video is represented by sampled timestamped frames; its first 30 seconds of audio are attached only with a Gemma 4 model when an audio stream is present. Direct audio input is currently Gemma 4-only.
- `publish_file` — transfers a finished file from the container to the host download area. The controller chooses a unique filename and never overwrites an existing download.
- Web search runs inside the container through `web-search "query"`; use `--json` for structured results and `--help` for its options. It supports engine, region, result-limit, timeout, and Safe Search settings.
- Browser automation runs inside the container through Python Playwright and the installed Chromium. It supports rendered pages, JavaScript, DOM inspection, interactions, forms, and downloads.
- HTTP clients are installed in the container. The `localagent_http` package exposes requests/httpx and an aiohttp wrapper whose `ClientSession` enables environment-proxy support by default.
- `get_tool_result` — retrieves sequential character ranges from a previously archived tool result using its returned `result_id`.

There are no model-facing `read_file`, `write_file`, `run_python`, Excel, Word, SQLite, copy, move, or delete tools. Those operations are performed with Bash and ordinary Linux/Python utilities inside the container. A small internal compatibility dispatcher can finish pending calls from older saved checkpoints, but legacy schemas are never sent to the model. `load_file_for_model` is the exception for direct model inspection of supported files, not a substitute for programmatic processing.

Shell state persists: `cd`, exported variables, shell functions, and files remain available in later terminal calls in the same conversation. Do not reset the working directory unless the task requires it. Install packages inside the container when necessary; the container is persistent for the life of the conversation.

### User attachments

Attachments are transferred into `/workspace/uploads/` in the conversation container through SWE-ReX file upload; their source attachment directories are not bind-mounted. The separate shared `/host-workspace` bind mount is documented in the sandbox section above. The original attachment remains available to the multimodal/document context as before. Files over 12 MB are not staged into the container by this integration.

### Publishing generated files

Call `publish_file` only when a finished artifact must be downloadable by the user. The tool copies the file to the host `workspace/downloads/` area and returns `/api/files/<relative-path>`. The current transfer limit is 8 MB per file. Ordinary file editing and copying inside the container should use Bash.

## Lifecycle and recovery

- One conversation ID maps to one container using label `localagent.task_id=<conversation_id>`.
- The mapping, container ID/name, host port, auth token, session name, status, and timestamps are stored in SQLite table `conversation_sandboxes`.
- Startup migration creates this table without deleting or rewriting existing conversation/task tables.
- On LocalAgent startup, `restore_all()` reconnects to recorded containers. It never creates a replacement for a missing recorded container.
- If a recorded container is stopped, Sandbox Manager starts that same container. Its filesystem remains; the old Bash process is gone, so session `main` is recreated.
- If the container is running, `create_session(main)` is attempted. `SessionExistsError` means the session survived and is accepted. `SessionDoesNotExistError` during execution triggers one controlled session recreation and retry.
- A missing container, invalid token, failed server health check, or other unrecoverable condition is reported; LocalAgent does not silently replace the container and lose files.
- Temporary HTTP/client disconnects and LocalAgent shutdown do not call `RemoteRuntime.close()` and do not remove the container.
- Deleting a conversation closes `main` if possible, removes only its labelled container, removes its sandbox DB record, and cleans up that conversation's managed uploaded attachments. The base image, other containers, and already-published downloads are not removed.

## Network and security model

The existing network configuration is intentionally not changed by this patch:

- Sandbox containers attach to `localagent-sandbox-internal-test`.
- The network is the already-configured internal bridge.
- Xray remains in its separate infrastructure container.
- Privoxy listens inside the sandbox on `127.0.0.1:8118` and forwards through the existing SOCKS5 route.
- Proxy environment variables are configured in the image; direct internet access is blocked by the existing network policy.
- The LocalAgent host connects to the published SWE-ReX port through loopback only.

Podman is controlled only by Sandbox Manager. The model receives no Podman CLI tool and no Podman socket. The container runs rootless on the host; container root is not host root. The server auth token is not stored in Podman labels, prompts, tool results, or ordinary LocalAgent logs. The container launcher scrubs the token from the initial environment and protects the SWE-ReX server process before it starts Bash sessions.

The full isolation boundary (host, LAN, router, other containers, process inspection, and all network paths) has not been independently penetration-tested by this patch. Do not treat the network or container security review as complete without running those tests on the target host.

## Resource limits and known limitations

- CPU limit: 2 cores.
- Memory limit: 4 GB.
- **Writable disk quota of 10 GB is not implemented.** No file-size scan is used as a substitute for a filesystem quota. A real quota requires separate host filesystem/storage configuration and is not applied automatically.
- Files in the container's writable Podman layer are deleted with the conversation/container. Files under the explicit `/host-workspace` bind mount live in `workspace/work` on the host and are not deleted with the container.
- A container that has been manually removed or whose storage is corrupted is not automatically reconstructed.
- The old Bubblewrap code remains inactive in `agent/tools.py` for rollback/reference. It is not registered as a model tool and is not selected automatically when `SANDBOX_ENABLED=false`.

## Tests

Run the test suite in the project virtual environment:

```bash
python -m py_compile agent/*.py container/swerex_server.py
pytest -q
```

Unit tests mock Podman and SWE-ReX. A full integration test requires the configured rootless Podman host, the prepared image, the existing internal network, and the SWE-ReX server. The full container lifecycle and network isolation must be tested on the target host before treating the migration as production-ready.

## Project layout

```text
agent/
  core.py                 # agent loop, planner integration, prompts
  db.py                   # SQLite schema and sandbox metadata
  sandbox_runtime.py      # SandboxRuntime protocol + SWE-ReX adapter
  sandbox_manager.py      # rootless Podman lifecycle and recovery
  tools.py                # model-facing tool catalog and dispatch
  planning.py             # planner and acceptance evidence
  web.py                  # web UI/API and published-file downloads
container/
  Containerfile            # prepared Ubuntu 24.04 image
  entrypoint.sh            # secure SWE-ReX supervisor entrypoint
  swerex_server.py         # token scrubbing, Privoxy and SWE-ReX startup
  requirements.txt         # container dependencies
  privoxy.conf             # existing proxy route
```


## Terminal waiting behavior

The model-facing computer interface is a single `terminal` tool. It does not see terminal pools, operation IDs, polling tools, or separate input/interrupt tools. While the persistent Bash command is running, an empty `command` means wait, ordinary text is sent as input to the foreground process, and Ctrl+C (`\\u0003`) interrupts it. After the command finishes, the next non-empty call starts a new Bash command.

- `TERMINAL_IDLE_TIMEOUT_SECONDS=10`: return control after this many seconds without new terminal output.
- `TERMINAL_MAX_WAIT_SECONDS=300`: maximum time one terminal tool call waits, even if output continues.

These values control when LocalAgent returns control to the LLM; they are not the remote command's hard execution timeout. The optional `timeout` argument remains a separate per-command hard timeout.


## Planning and context lifecycle (0.7.2)

- The production LLM chooses Direct or Plan. Repeated tool failures trigger a second model decision; they do not automatically force Plan.
- New plan stages use only `pending`, `in_progress`, `completed`, `skipped`, and `question`. A skipped stage is resolved. A question pauses the task and is resumed when the user replies. The final answer is outside the plan.
- Ordinary stages do not require a fresh tool call solely to become completed. A declared machine-checkable `acceptance_check` remains a hard gate.
- Context budgeting counts the active tool schema and reserves output tokens. Around 50%, only exact event-ID duplicates are removed from the working context; around 60%, the controller records a deterministic growth forecast; around 70%, the LLM compacts a factual summary. Original messages and tool results remain archived.
- `load_file_for_model` estimates text and multimodal context cost before attaching content. If Ollama still reports context overflow, the failed payload is removed from the next model request and a clear retry/splitting instruction is returned.


## Model file input

load_file_for_model and ask_model share the same container-file preparation path. The default text allowance is approximately 32k input tokens (a conservative 80,000-character ceiling); a default load refuses oversized text instead of silently returning a prefix. Explicit start/length reads remain available for deliberate sequential inspection. The context guard also counts previously attached media and reserves space for the model response.

- Images: PNG, JPEG, WebP and GIF; 12 MB per file.
- Audio: mono 16 kHz, 32-bit float WAV after FFmpeg normalization; maximum 30 seconds and 12 MB source file.
- Video: maximum 60 seconds and 100 MB source file; sampled at up to 1 frame/second, no more than 60 frames, normalized to compact 320 px JPEG frames. For Gemma 4, up to 30 seconds of soundtrack is separately normalized as audio.
- Documents: up to 50 MB. PDF text is extracted and pages containing scans, embedded images or substantial vector graphics are rendered as visual attachments. DOCX/XLSX text is extracted and embedded images are attached when available. Visual extraction is capped at 12 images per document; all text and visuals share the 32k-token input budget. Native XLSX chart objects are currently reported but not rasterized, so use terminal/LibreOffice for their visual rendering.

Browser uploads are stored unchanged under workspace/work and are exposed inside the conversation container as /host-workspace/<filename>. The user message tells the agent to load each file with load_file_for_model; the original prompt is included only once, and the file is not also duplicated as a direct attachment. Unsupported formats and over-budget files are reported explicitly.

Ollama adapter compatibility is intentionally not probed by this implementation; media transport is based on the configured model-family assumptions and should be validated separately.
