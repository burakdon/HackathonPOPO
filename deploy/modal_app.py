"""NeuroCue patient console on Modal (https://modal.com), an alternative host for the static console.

The hosted demo now lives on GitHub Pages (https://burakdon.github.io/neurocue/, published by
.github/workflows/deploy-pages.yml), which serves the same directory for free. This app is kept in
case a server is wanted; it draws on Modal credits while a container is up.

    uv tool install modal            # once per machine
    modal setup                      # once per machine: browser login
    modal deploy deploy/modal_app.py # publish or update; prints the URL (https://burakdonbekci--neurocue-web.modal.run)

The hosted console needs no live compute: the six patients it offers, with their naming responses,
therapy courses and DITTO results, are precomputed by scripts/build_demo_patients.py into
neurocue/demo/static/patients/. So the image is a slim Python with FastAPI serving that one
directory: no torch, no engine, a cold start of a few seconds. The full live server
(neurocue/demo/server.py, any patient, live DITTO) still runs locally with ./scripts/run_demo.sh.
"""
from pathlib import Path

import modal

STATIC = Path(__file__).resolve().parents[1] / "neurocue" / "demo" / "static"
REMOTE = "/root/static"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi==0.115.12")
    .add_local_dir(STATIC, remote_path=REMOTE)
)

app = modal.App("neurocue", image=image)


@app.function(cpu=0.25, memory=256, scaledown_window=5 * 60, timeout=60)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import hashlib

    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from starlette.datastructures import Headers
    from starlette.responses import FileResponse

    class ContentStatic(StaticFiles):
        """Modal mounts files with a 1970 mtime, and Starlette's ETag is mtime + size, so an edit that
        keeps a file's size (an age 52 -> 23) would be answered 304 and browsers would keep the old file.
        Tag each file by its content instead, and ask browsers to revalidate on every load."""
        _tags: dict[str, str] = {}

        def file_response(self, full_path, stat_result, scope, status_code=200):
            path = str(full_path)
            if path not in self._tags:
                with open(path, "rb") as f:
                    self._tags[path] = '"' + hashlib.md5(f.read()).hexdigest() + '"'
            response = FileResponse(full_path, status_code=status_code, stat_result=stat_result)
            response.headers["etag"] = self._tags[path]
            response.headers["cache-control"] = "no-cache"
            del response.headers["last-modified"]
            if self.is_not_modified(response.headers, Headers(scope=scope)):
                from starlette.responses import Response
                return Response(status_code=304, headers={"etag": self._tags[path], "cache-control": "no-cache"})
            return response

    site = FastAPI(title="NeuroCue patient console", docs_url=None, redoc_url=None, openapi_url=None)
    site.mount("/", ContentStatic(directory=REMOTE, html=True), name="static")
    return site
