"""Record a captioned walkthrough of the patient console as a video.

  uv run python -m neurocue.demo.record_demo [--url http://localhost:8765] [--out results/demo]

Drives the live console in headless Chrome (Playwright, system Chrome), burns a caption bar into
the frame, saves <out>/neurocue_demo.webm and .mp4 (bundled ffmpeg) plus an .srt of the captions.
The server must already be running; the trainer/override steps run only if the trained policy is
loaded there.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

CAPTION_CSS = """
#neurocue-cap { position: fixed; left: 50%; bottom: 22px; transform: translateX(-50%); max-width: 86%;
  background: rgba(10, 14, 13, .86); color: #fff; font: 600 22px/1.35 -apple-system, "Segoe UI", Inter, sans-serif;
  padding: 12px 20px; border-radius: 12px; z-index: 99999; text-align: center; box-shadow: 0 6px 24px rgba(0,0,0,.35);
  transition: opacity .25s; }
#neurocue-cap:empty { opacity: 0; }
"""


class Recorder:
    def __init__(self, page, t0: float) -> None:
        self.page, self.t0, self.srt = page, t0, []
        self.page.add_style_tag(content=CAPTION_CSS)
        self.page.evaluate("() => { const d = document.createElement('div'); d.id = 'neurocue-cap'; document.body.appendChild(d); }")
        self._open = None

    def cap(self, text: str) -> None:
        now = time.time() - self.t0
        if self._open:
            self.srt.append((self._open[0], now, self._open[1]))
        self._open = (now, text)
        self.page.evaluate("t => { document.getElementById('neurocue-cap').textContent = t; }", text)

    def wait(self, s: float) -> None:
        self.page.wait_for_timeout(int(s * 1000))

    def close(self) -> None:
        if self._open:
            self.srt.append((self._open[0], time.time() - self.t0, self._open[1]))

    def write_srt(self, path: Path) -> None:
        def ts(x: float) -> str:
            h, r = divmod(x, 3600); m, s = divmod(r, 60)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"
        path.write_text("\n".join(f"{i + 1}\n{ts(a)} --> {ts(b)}\n{t}\n" for i, (a, b, t) in enumerate(self.srt)))


def set_range(page, selector: str, value: float) -> None:
    page.evaluate("([s, v]) => { const e = document.querySelector(s); e.value = v; e.dispatchEvent(new Event('input')); }", [selector, value])


def latest_response(page) -> dict:
    return page.evaluate("""() => {
        const r = document.querySelector('#feed .r'); if (!r) return null;
        const c = r.children; return {target: c[1].textContent, said: c[2].textContent.trim(), cat: c[3].textContent};
    }""")


def name_n(rec: Recorder, page, n: int, pause: float = 2.4) -> None:
    for _ in range(n):
        page.get_by_role("button", name="Name it").click()
        page.wait_for_function("() => document.querySelector('#said').textContent !== '…'", timeout=20000)
        rec.wait(0.2)
        r = latest_response(page)
        if r:
            label = {"correct": "correct", "semantic": "semantic error", "formal": "sound-related word", "mixed": "mixed error",
                     "unrelated": "unrelated word", "nonword": "phonological nonword", "omission": "no response"}.get(r["cat"], r["cat"])
            rec.cap(f"{r['target']} → “{r['said']}”  ({label})")
        rec.wait(pause)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--first", default="55,phonological,3", help="aq,lesion,seed of the opening patient")
    ap.add_argument("--contrast", default="50,semantic,5", help="aq,lesion,seed of the contrast patient")
    args = ap.parse_args()
    a1, l1, s1 = args.first.split(","); a2, l2, s2 = args.contrast.split(",")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vdir = out / "_video"; shutil.rmtree(vdir, ignore_errors=True); vdir.mkdir()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900}, record_video_dir=str(vdir),
                                  record_video_size={"width": 1280, "height": 900}, color_scheme="light")
        page = ctx.new_page()
        page.goto(args.url, wait_until="networkidle")
        t0 = time.time()
        rec = Recorder(page, t0)
        rec.cap("NeuroCue: a synthetic aphasia patient you can run experiments on. Everything on screen is the validated engine, live.")
        rec.wait(4)

        # 1. a moderate, phonological patient
        rec.cap(f"Choosing who to listen to: moderate severity (AQ {a1}), a {l1} lesion.")
        set_range(page, "#aq", int(a1)); page.get_by_role("button", name=l1, exact=True).click(); page.fill("#seed", s1)
        rec.wait(1.5)
        page.get_by_role("button", name="New patient").click(); rec.wait(2)
        page.locator("#speak").uncheck()
        rec.cap("The patient is the Dell model of word retrieval with two lesion weights drawn from a prior fitted to real patients.")
        rec.wait(3.5)
        rec.cap("Now the patient names pictures. The avatar pauses for as long as the response took, then mouths the phonemes it actually produced.")
        rec.wait(3)
        name_n(rec, page, 8)

        # 2. diagnose
        page.locator("#diagnose").scroll_into_view_if_needed(); rec.wait(0.5)
        rec.cap("Diagnosis: the fitter's posterior over the two lesion weights, from those eight answers alone. Bright = likely.")
        rec.wait(4)
        page.get_by_role("button", name="Reveal true lesion").click(); rec.wait(0.8)
        rec.cap("Revealed: the true lesion (orange) sits inside the bright region. This is the fitter that reproduced Dell's published patients.")
        rec.wait(5)

        # 3. contrast patient
        page.locator("#patient").scroll_into_view_if_needed(); rec.wait(0.5)
        rec.cap(f"Same pictures, different person: AQ {a2}, a {l2} lesion. Listen to what changes.")
        set_range(page, "#aq", int(a2)); page.get_by_role("button", name=l2, exact=True).click(); page.fill("#seed", s2)
        rec.wait(1.5)
        page.get_by_role("button", name="New patient").click(); rec.wait(2)
        page.locator("#speak").uncheck()
        name_n(rec, page, 6, pause=2.2)

        # 4. trainer
        page.evaluate("() => window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})"); rec.wait(1.2)
        ppo = page.evaluate("() => (document.querySelector('#runnote').textContent || '').includes('trained policy, same')")
        rec.cap("A trainer treats the patient: four sessions over ten days on 20 hard words, judged on what is still named a month later.")
        rec.wait(3)
        page.get_by_role("button", name="Run a four-session course").click()
        page.wait_for_function("() => (document.querySelector('#summary').textContent || '').startsWith('Words')", timeout=120000)
        page.evaluate("() => window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})"); rec.wait(1.0)
        rec.cap(page.locator("#summary").inner_text())
        rec.wait(6)
        if ppo:
            idx = page.evaluate("() => { const cells = [...document.querySelectorAll('#strip div')]; return cells.findIndex((d, i) => i > 2 && d.style.background.includes('nonword')); }")
            if idx is not None and idx >= 0:
                rec.cap("The clinician disagrees with one choice: after this error she would have given the first sound.")
                page.locator("#strip div").nth(idx).click(); rec.wait(1.5)
                page.select_option("#ocue", "phonemic_initial"); rec.wait(1)
                rec.cap("Apply override: DITTO retrains the policy on this single demonstration, anchored to the original.")
                page.get_by_role("button", name="Apply override (DITTO)").click()
                page.wait_for_function("() => (document.querySelector('#ostat').textContent || '') === 'done'", timeout=180000)
                page.evaluate("() => window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})"); rec.wait(1.0)
                rec.cap(page.locator("#summary").inner_text())
                rec.wait(7)
        rec.cap("Validated layer by layer: published model fits, 1,074 real responses (328 clinician-coded), a real therapy study, cued-naming data.")
        rec.wait(5)
        rec.cap("")
        rec.wait(1)
        rec.close()
        ctx.close(); browser.close()

    video = next(vdir.glob("*.webm"))
    webm = out / "neurocue_demo.webm"; shutil.move(str(video), webm)
    rec.write_srt(out / "neurocue_demo.srt")
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        mp4 = out / "neurocue_demo.mp4"
        subprocess.run([ff, "-y", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "25", str(mp4)],
                       check=True, capture_output=True)
        print(f"video -> {mp4}")
    except Exception as e:   # noqa: BLE001
        print(f"mp4 conversion skipped: {e}")
    print(f"video -> {webm}\ncaptions -> {out / 'neurocue_demo.srt'}")


if __name__ == "__main__":
    main()
