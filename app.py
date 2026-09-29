"""
SPARK AI - Application Supervisor

Supports:
1. Streamlit environments (such as Hugging Face Spaces where `streamlit run app.py` is invoked):
   Starts FastAPI backend in a background process if not already running,
   then renders frontend/app.py directly in the active Streamlit session.
2. Local execution (`python app.py`):
   Starts FastAPI backend, waits for /health, then launches Streamlit.
"""
from __future__ import annotations

import atexit
import os
import runpy
import subprocess
import sys
import time

_API_PROC: subprocess.Popen | None = None


def _is_backend_healthy(port: int) -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def _terminate_api() -> None:
    global _API_PROC
    if _API_PROC is not None and _API_PROC.poll() is None:
        _API_PROC.terminate()
        try:
            _API_PROC.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _API_PROC.kill()


def _ensure_backend_running(root: str, api_port: int) -> None:
    global _API_PROC
    if _is_backend_healthy(api_port):
        return

    if os.environ.get("SKIP_EMBEDDED_FASTAPI", "").lower() in ("1", "true", "yes"):
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
        if _is_backend_healthy(api_port):
            break
        time.sleep(1)
    else:
        print(
            "WARNING: FastAPI did not become healthy in time; UI may show backend offline.",
            file=sys.stderr,
        )


def _is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    api_port = int(os.environ.get("FASTAPI_INTERNAL_PORT", "7861"))
    os.environ.setdefault("BACKEND_URL", f"http://127.0.0.1:{api_port}")

    _ensure_backend_running(root, api_port)

    frontend = os.path.join(root, "frontend", "app.py")

    if _is_running_in_streamlit():
        # Running inside Streamlit server (e.g. `streamlit run app.py`)
        # Render frontend UI directly into the active Streamlit app
        runpy.run_path(frontend, run_name="__main__")
    else:
        # Running via `python app.py` (local command line)
        # Launch Streamlit subprocess
        st_port = os.environ.get("PORT", "7860")
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


if __name__ == "__main__":
    main()
