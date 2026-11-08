"""Mock server for the API contract: serves docs/contract_examples/ at the real endpoint paths.

For frontend work before the backend exists. Any ID in a path is accepted. The practice endpoints
(`POST /sessions/{id}/recordings` and `/no-answer`) step through the scripted session in order;
starting a session (`POST /patients/{id}/sessions`) rewinds it. Images come from frontend/images;
hint clips and patient recordings are a short placeholder tone. CORS is open, so product/index.html
can call it straight from a file:// page.

Run:  uv run uvicorn service.contract_mock:app --port 8001
Then: http://localhost:8001/product/index.html?api#patient   (tablet in server mode)
"""
from __future__ import annotations

import io
import json
import math
import re
import struct
import wave

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from core.config import resolve

EXAMPLES = resolve("docs/contract_examples")


def _tone(seconds: float = 0.6, hz: float = 440.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * hz * i / rate)
                                                     * min(1, i / 800, (n - i) / 800))) for i in range(n)))
    return buf.getvalue()


def _pattern(path: str) -> re.Pattern:
    return re.compile("^" + re.sub(r"\\\{[^}]+\\\}", "[^/]+", re.escape(path)) + "$")


def create_mock_app() -> FastAPI:
    index = json.loads((EXAMPLES / "index.json").read_text())
    tone = _tone()
    script = list(index["practice_script"])
    state = {"step": 0}
    # first success sample per (method, path); error examples and script steps are not routed directly
    routes: list[tuple[str, re.Pattern, dict]] = []
    seen = set()
    for e in index["endpoints"]:
        key = (e["method"], e["path"])
        if e["status"] >= 400 or e["file"] in script or key in seen:
            continue
        seen.add(key)
        routes.append((e["method"], _pattern(e["path"]), e))

    app = FastAPI(title="CueAgain contract mock", description=__doc__)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.mount("/media/images", StaticFiles(directory=resolve("frontend/images")), name="images")
    # the product page itself, so it runs on http://localhost (browsers allow the microphone there)
    app.mount("/product", StaticFiles(directory=resolve("product"), html=True), name="product")

    def load(rel: str):
        return json.loads((EXAMPLES / rel).read_text())

    @app.get("/media/hints/{exercise_id}/{clip}")
    def hint_clip(exercise_id: str, clip: str) -> Response:
        return Response(tone, media_type="audio/wav", headers={"X-Mock": "placeholder tone"})

    @app.get("/attempts/{attempt_id}/audio")
    def attempt_audio(attempt_id: str) -> Response:
        return Response(tone, media_type="audio/wav", headers={"X-Mock": "placeholder tone"})

    @app.post("/mock/reset")
    def reset() -> dict:
        state["step"] = 0
        return {"step": 0, "of": len(script)}

    @app.post("/sessions/{session_id}/recordings")
    @app.post("/sessions/{session_id}/no-answer")
    async def practice_step(session_id: str, request: Request) -> JSONResponse:
        k = state["step"]
        if k >= len(script):
            return JSONResponse({"detail": "the scripted session is complete; start a new session to rewind"},
                                status_code=409)
        state["step"] += 1
        endpoint = request.url.path.rsplit("/", 1)[-1]
        expected = "no-answer" if "_no-answer_" in script[k] else "recordings"
        return JSONResponse(load(script[k]), headers={"X-Mock-Step": f"{k + 1}/{len(script)}",
                                                      "X-Mock-Expected-Endpoint": expected,
                                                      "X-Mock-Endpoint-Matches": str(endpoint == expected).lower()})

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT"])
    async def any_route(path: str, request: Request) -> JSONResponse:
        url = "/" + path
        for method, pat, e in routes:
            if method == request.method and pat.match(url):
                if e["path"] == "/patients/{patient_id}/sessions":
                    state["step"] = 0
                return JSONResponse(load(e["file"]), status_code=e["status"], headers={"X-Mock-File": e["file"]})
        return JSONResponse({"detail": f"no sample for {request.method} {url}"}, status_code=404)

    return app


app = create_mock_app()
