from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class Settings:
    # Provider is a deployment default; the selected model remains a UI/DB setting and is pinned per task.
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    llama_cpp_url: str = os.getenv("LLAMA_CPP_URL", "http://127.0.0.1:8081/v1")
    llama_cpp_model: str = os.getenv("LLAMA_CPP_MODEL", "/home/buzuck/AI/Bonsai2/models/27B/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
    llama_cpp_aux_model: str = os.getenv("LLAMA_CPP_AUX_MODEL", "gemma4-12b")
    llama_cpp_context: int = int(os.getenv("LLAMA_CPP_CONTEXT", "32768"))
    llama_cpp_timeout: int = int(os.getenv("LLAMA_CPP_TIMEOUT", "600"))
    model: str = os.getenv("OLLAMA_MODEL", "ornith-1.5:9b")
    # Conservative budget used until Ollama reports the loaded model's
    # actual context through /api/ps. Mirror OLLAMA_CONTEXT_LENGTH here when
    # that server-level setting is used; OLLAMA_NUM_CTX is legacy and is not
    # evidence of the context allocated by Ollama.
    ollama_context_fallback: int = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "4096"))
    # Legacy LocalAgent-only setting retained for compatibility; do not use it
    # as evidence of Ollama's allocated runtime context.
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "65536"))
    keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
    # Legacy setting retained for compatibility; chat uses the explicit inactivity timeouts below.
    request_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    ollama_connect_timeout: float = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "10"))
    ollama_first_token_timeout: float = float(os.getenv("OLLAMA_FIRST_TOKEN_TIMEOUT", "600"))
    ollama_stream_idle_timeout: float = float(os.getenv("OLLAMA_STREAM_IDLE_TIMEOUT", "300"))
    database_path: Path = Path(os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "agent.sqlite3")))
    workspace_path: Path = Path(os.getenv("WORKSPACE_PATH", str(PROJECT_ROOT / "workspace"))).resolve()
    host: str = os.getenv("WEB_HOST", "127.0.0.1")
    port: int = int(os.getenv("WEB_PORT", "8765"))
    max_tool_rounds: int = int(os.getenv("MAX_TOOL_ROUNDS", "80"))
    loop_warning_threshold: int = int(os.getenv("LOOP_WARNING_THRESHOLD", "3"))
    loop_pause_threshold: int = int(os.getenv("LOOP_PAUSE_THRESHOLD", "6"))
    auto_resume_tasks: bool = _bool("AUTO_RESUME_TASKS", False)
    # Conversation-scoped rootless Podman + SWE-ReX sandbox.
    sandbox_enabled: bool = _bool("SANDBOX_ENABLED", True)
    podman_path: str = os.getenv("PODMAN_PATH", "podman")
    sandbox_image: str = os.getenv("SANDBOX_IMAGE", "localhost/localagent-sandbox:ubuntu24.04")
    sandbox_network: str = os.getenv("SANDBOX_NETWORK", "localagent-sandbox-internal-test")
    sandbox_cpus: float = float(os.getenv("SANDBOX_CPUS", "2"))
    sandbox_memory: str = os.getenv("SANDBOX_MEMORY", "4g")
    sandbox_container_port: int = int(os.getenv("SANDBOX_CONTAINER_PORT", "8000"))
    sandbox_proxy_required: bool = _bool("SANDBOX_PROXY_REQUIRED", True)
    xray_proxy_container: str = os.getenv("XRAY_PROXY_CONTAINER", "localagent-xray-proxy")
    debug_llm_trace: bool = _bool("DEBUG_LLM_TRACE", False)
    debug_llm_thinking: bool = _bool("DEBUG_LLM_THINKING", False)
    swerex_timeout: float = float(os.getenv("SWEREX_TIMEOUT", "10"))
    # Internal observation window before a separate diagnostic review.
    # Silence alone never interrupts a command or calls the main LLM.
    terminal_idle_timeout_seconds: float = float(os.getenv("TERMINAL_IDLE_TIMEOUT_SECONDS", "300"))
    terminal_max_wait_seconds: float = float(os.getenv("TERMINAL_MAX_WAIT_SECONDS", "300"))
    terminal_poll_interval_seconds: float = float(os.getenv("TERMINAL_POLL_INTERVAL_SECONDS", "2"))

settings = Settings()
settings.database_path.parent.mkdir(parents=True, exist_ok=True)
settings.workspace_path.mkdir(parents=True, exist_ok=True)
