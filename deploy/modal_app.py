"""NeuroCue patient console on Modal (https://modal.com): serverless containers, public HTTPS URL.

    uv tool install modal            # once per machine
    modal setup                      # once per machine: browser login
    modal deploy deploy/modal_app.py # publish or update; prints the URL (https://burakdonbekci--neurocue-web.modal.run)

The image is built from pyproject.toml and uv.lock, so it is the same dependency set as the local
environment (on Linux torch comes from the CPU-only index, see pyproject.toml). The repository is
mounted at /root/neurocue with the .dockerignore rules, so transcripts and other ignored files
never leave the machine. The FastAPI app in neurocue/demo/server.py loads the engine in its startup
event; the first request after a cold start waits for that (about a minute), later requests are
instant. The container stays up for 20 minutes after the last request.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/neurocue"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(uv_project_dir=str(ROOT), extra_options="--no-dev")
    .env({"PYTHONUNBUFFERED": "1", "NLTK_DATA": f"{REMOTE}/data/cache/nltk_data"})
    .add_local_dir(ROOT, remote_path=REMOTE, ignore=modal.FilePatternMatcher.from_file(ROOT / ".dockerignore"))
)

app = modal.App("neurocue", image=image)


@app.function(cpu=2, memory=3072, scaledown_window=20 * 60, timeout=600, startup_timeout=600)
@modal.concurrent(max_inputs=40)
@modal.asgi_app()
def web():
    import os
    import sys

    os.chdir(REMOTE)
    sys.path.insert(0, REMOTE)
    from neurocue.demo.server import app as fastapi_app

    return fastapi_app
