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
        before = rows(page)
        page.get_by_role("button", name="Name it").click()
        page.wait_for_function("b => document.querySelectorAll('#feed .r').length > b", arg=before, timeout=30000)
        rec.wait(0.2)
        r = latest_response(page)
        if r:
            label = {"correct": "correct", "semantic": "semantic error", "formal": "sound-related word", "mixed": "mixed error",
                     "unrelated": "unrelated word", "nonword": "phonological nonword", "omission": "no response"}.get(r["cat"], r["cat"])
            rec.cap(f"{r['target']} → “{r['said']}”  ({label})")
        rec.wait(pause)


def rows(page) -> int:
    return page.evaluate("() => document.querySelectorAll('#feed .r').length")


def name_batch(page, n: int) -> None:
    """Name n pictures at once through the page's batch control and wait for the rows to land."""
    before = rows(page)
    page.fill("#nbatch", str(n)); page.get_by_role("button", name="Go", exact=True).click()
    page.wait_for_function("b => document.querySelectorAll('#feed .r').length >= b", arg=before + n, timeout=90000)


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
        rec.cap("Marisol, 23, is the Dell model of word retrieval: two lesion weights drawn from a prior fitted to real PSST patients.")
        rec.wait(3.5)
        page.locator("#naming").scroll_into_view_if_needed(); rec.wait(0.5)
        rec.cap("She names pictures. The avatar mouths the phonemes she actually produced, with the false starts and carrier phrases real speakers use.")
        rec.wait(3.5)
        name_n(rec, page, 6)
        rec.cap("Or skip the listening: fourteen more answers at once, from the same patient.")
        name_batch(page, 14)
        rec.wait(2.5)

        # 2. diagnose
        page.locator("#diagnose").scroll_into_view_if_needed(); rec.wait(0.5)
        rec.cap("Diagnosis from those 20 answers alone: the fitter's posterior over the two lesion weights. Bright = likely.")
        rec.wait(4)
        page.get_by_role("button", name="Reveal true lesion").click(); rec.wait(0.8)
        rec.cap("Revealed: the true lesion sits inside the bright region. The same fitter reproduced Dell's published patients.")
        rec.wait(5)

        # 3. contrast patient
        page.locator("#patient").scroll_into_view_if_needed(); rec.wait(0.5)
        rec.cap(f"Same pictures, different person: AQ {a2}, a {l2} lesion. Listen to what changes.")
        set_range(page, "#aq", int(a2)); page.get_by_role("button", name=l2, exact=True).click(); page.fill("#seed", s2)
        rec.wait(1.5)
        page.get_by_role("button", name="New patient").click(); rec.wait(2)
        page.locator("#speak").uncheck()
        page.locator("#naming").scroll_into_view_if_needed(); rec.wait(0.3)
        name_n(rec, page, 5, pause=2.2)
        name_batch(page, 15)
        page.locator("#diagnose").scroll_into_view_if_needed(); rec.wait(0.5)
        rec.cap("Her errors are whole-word substitutions rather than nonwords, and the posterior moves to the semantic side.")
        rec.wait(4.5)

        # 4. two therapists
        page.locator("#trainer").scroll_into_view_if_needed(); rec.wait(0.8)
        rec.cap("Two therapists treat identical copies of her: the clinic's cueing hierarchy, and a policy trained on this engine. Four sessions over ten days, 20 hard words.")
        rec.wait(4.5)
        page.get_by_role("button", name="Run a four-session course").click()
        page.wait_for_function("() => ['Done.', 'Failed.'].includes((document.querySelector('#runstate').textContent || '').trim())", timeout=240000)
        rec.wait(1.0)
        summary = page.locator("#summary").inner_text()
        rec.cap(summary.split("Both lines")[0].strip()[:260])
        rec.wait(7)
        rec.cap("The strip is the trained policy's course, trial by trial: which word, which cue. Nobody told it to space retrieval; it learned that from the engine.")
        rec.wait(5.5)

        # 5. where the physician comes in
        page.locator("#teach").scroll_into_view_if_needed(); rec.wait(0.8)
        rec.cap("Where the physician comes in. One correction is one demonstration; DITTO folds it into the policy, anchored to what the policy already knew.")
        rec.wait(4)
        page.get_by_role("button", name="Play one correction").click()
        rec.wait(5.5)
        rec.cap("Over months the corrections compound: more of them, applied consistently, move the policy toward the clinician's own practice. An illustration, not data.")
        set_range(page, "#slCorr", 8); rec.wait(1.6); set_range(page, "#slCons", 70); rec.wait(1.6); set_range(page, "#slCons", 97); rec.wait(2.8)
        teachable = page.evaluate("() => !document.querySelector('#tapply').disabled")
        if teachable:
            # teach the rule for the error kind this course produced most, so the held-out check has moments to judge
            counts = page.evaluate("() => { const r = taughtRun(); const c = {}; r.trials.forEach(t => { c[t.prev] = (c[t.prev] || 0) + 1; }); return c; }")
            opts = page.evaluate("() => [...document.querySelector('#tctx').options].map(o => o.value)")
            cands = [c for c in ("nonword", "semantic", "formal", "mixed", "unrelated") if c in opts]
            teach_ctx = max(cands, key=lambda c: counts.get(c, 0)) if cands else "any_error"
            label = {"nonword": "a nonword", "semantic": "a meaning error", "formal": "a sound-related word", "mixed": "a mixed error",
                     "unrelated": "an unrelated word"}.get(teach_ctx, "any error")
            page.select_option("#tctx", teach_ctx); page.select_option("#trule", "repeat"); page.select_option("#tcue", "phonemic_initial")
            rec.cap(f"Live, on her recorded course: the clinician teaches one rule. After {label}, retry the word with its first sound.")
            rec.wait(3.5)
            page.locator("#tapply").click()
            page.wait_for_function("() => (document.querySelector('#tstat').textContent || '') === 'done'", timeout=300000)
            page.locator("#ditto").scroll_into_view_if_needed(); rec.wait(1.0)
            chips = page.evaluate("() => [...document.querySelectorAll('#osummary .chip')].map(e => e.textContent.trim()).join('  ·  ')")
            rec.cap(chips[:240] or "Taught.")
            rec.wait(7)
            rec.cap("Judged on two things only: adoption where the rule applies, containment everywhere else. The policy took the clinician's rule, and nothing more.")
            rec.wait(5.5)
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
