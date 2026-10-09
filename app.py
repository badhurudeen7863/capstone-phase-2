"""
Hugging Face Space entrypoint for ExpenseIQ.

A single free HF Space serves the whole app behind one public URL:

    https://<hf-user>-<space-name>.hf.space

This launcher starts the FastAPI backend in a background thread (in-process
uvicorn bound to 127.0.0.1) and then renders the Streamlit dashboard in the
foreground. The dashboard calls the API server-side via API_BASE_URL; the
browser only ever talks to Streamlit, so one public URL is enough.

Environment variables:
    API_PORT   port for the in-process API (default 8000; HF Spaces use 7860
               for Streamlit itself, so the API must stay on a different port)
"""
from __future__ import annotations

import os
import runpy
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

API_PORT = int(os.environ.get("API_PORT", "8000"))
# frontend/client.py reads this at import time - set it before importing anything.
os.environ["API_BASE_URL"] = f"http://127.0.0.1:{API_PORT}"


def _api_ready() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", API_PORT)) == 0


def _run_backend() -> None:
    # Seed the demo database once (scripts/seed_db.py is idempotent: existing
    # rows are skipped), then serve the API. Runs in a daemon thread.
    subprocess.run([sys.executable, "scripts/seed_db.py"], cwd=ROOT, check=False)
    import uvicorn

    from backend.main import app as api_app

    config = uvicorn.Config(api_app, host="127.0.0.1", port=API_PORT, log_level="info")
    uvicorn.Server(config).run()


if not _api_ready():
    threading.Thread(target=_run_backend, name="expenseiq-api", daemon=True).start()
    # Block the first render until the API is up, so the login page works
    # immediately instead of showing a connection error.
    deadline = time.time() + 180
    while not _api_ready() and time.time() < deadline:
        time.sleep(1)

# Execute the dashboard script fresh on every Streamlit rerun (runpy avoids
# the module-caching problem a plain `import frontend.app` would cause).
runpy.run_path(str(ROOT / "frontend" / "app.py"), run_name="__main__")
