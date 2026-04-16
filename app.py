"""
Hugging Face Spaces / local all-in-one launcher.

Starts FastAPI (backend) in a subprocess, waits for /health, then runs Streamlit.
Set SKIP_EMBEDDED_FASTAPI=1 if you run uvicorn separately and only want the UI.

Environment:
  PORT              — Streamlit port (Hugging Face sets this, often 7860)
  FASTAPI_INTERNAL_PORT — API port (default 7861)
  BACKEND_URL       — Auto-set to match the API unless already set
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time

_API_PROC: subprocess.Popen | None = None


def _terminate_api() -> None:
    global _API_PROC
    if _API_PROC is not None and _API_PROC.poll() is None:
        _API_PROC.terminate()
        try:
            _API_PROC.wait(timeout=12)
        except subprocess.TimeoutExpired:
            _API_PROC.kill()


def main() -> None:
    global _API_PROC
    root = os.path.dirname(os.path.abspath(__file__))
    api_port = int(os.environ.get("FASTAPI_INTERNAL_PORT", "7861"))
    os.environ.setdefault("BACKEND_URL", f"http://127.0.0.1:{api_port}")

    if os.environ.get("SKIP_EMBEDDED_FASTAPI", "").lower() in ("1", "true", "yes"):
        _run_streamlit(root)
        return

    _API_PROC = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            f"--port={api_port}",
        ],
        cwd=root,
    )
    atexit.register(_terminate_api)

    for _ in range(45):
        try:
            import urllib.error
            import urllib.request

            urllib.request.urlopen(f"http://127.0.0.1:{api_port}/health", timeout=1)
            break
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    else:
        print(
            "WARNING: FastAPI did not become healthy in time; UI may show backend offline.",
            file=sys.stderr,
        )

    _run_streamlit(root)


def _run_streamlit(root: str) -> None:
    st_port = os.environ.get("PORT", "7860")
    frontend = os.path.join(root, "frontend", "app.py")
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                frontend,
                "--server.address",
                "0.0.0.0",
                "--server.port",
                st_port,
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
                "--server.enableCORS",
                "false",
                "--server.enableXsrfProtection",
                "false",
            ],
            cwd=root,
        )
    )


if __name__ == "__main__":
    main()
