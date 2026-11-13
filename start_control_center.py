"""Start local Ollama and the Gemma dashboard, then print their URLs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
OLLAMA_URL = "http://127.0.0.1:11434"
DASHBOARD_URL = "http://127.0.0.1:8765"
WINDOWS_FLAGS = (
    getattr(subprocess, "DETACHED_PROCESS", 0)
    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    | getattr(subprocess, "CREATE_NO_WINDOW", 0)
)


def read_json(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def start_process(command: list[str], name: str) -> None:
    log_dir = Path(tempfile.gettempdir())
    stdout = (log_dir / f"gemma-control-center-{name}.log").open("ab")
    stderr = (log_dir / f"gemma-control-center-{name}-error.log").open("ab")
    subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        creationflags=WINDOWS_FLAGS if os.name == "nt" else 0,
        close_fds=os.name != "nt",
    )
    # Keep the file objects open in this process while detached children inherit them.
    start_process._handles = getattr(start_process, "_handles", []) + [stdout, stderr]


def wait_for(url: str, seconds: int) -> dict | None:
    for _ in range(seconds):
        value = read_json(url)
        if value is not None:
            return value
        time.sleep(1)
    return None


def main() -> int:
    ollama = shutil.which("ollama")
    if not ollama and os.name == "nt":
        local_ollama = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if local_ollama.is_file():
            ollama = str(local_ollama)
    if not ollama:
        print("Ollama was not found. Install it or add ollama.exe to PATH.", file=sys.stderr)
        return 1

    tags_url = f"{OLLAMA_URL}/api/tags"
    if read_json(tags_url) is None:
        start_process([ollama, "serve"], "ollama")
    models = wait_for(tags_url, 60)
    if models is None:
        print(f"Ollama did not become ready at {OLLAMA_URL}.", file=sys.stderr)
        print(f"See {Path(tempfile.gettempdir()) / 'gemma-control-center-ollama-error.log'}", file=sys.stderr)
        return 1
    model_names = [model.get("name") for model in models.get("models", [])]
    if "gemma3:4b" not in model_names:
        print("Warning: gemma3:4b is not installed in Ollama.", file=sys.stderr)

    dashboard_probe = f"{DASHBOARD_URL}/api/models"
    if read_json(dashboard_probe) is None:
        start_process([sys.executable, str(ROOT / "visualizer" / "server.py")], "dashboard")
    if wait_for(dashboard_probe, 15) is None:
        print(f"Dashboard did not become ready at {DASHBOARD_URL}.", file=sys.stderr)
        print(f"See {Path(tempfile.gettempdir()) / 'gemma-control-center-dashboard-error.log'}", file=sys.stderr)
        return 1

    print(f"Ollama is ready: {OLLAMA_URL}")
    print(f"Control center is ready: {DASHBOARD_URL}")
    print("Open the control center URL in your browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
