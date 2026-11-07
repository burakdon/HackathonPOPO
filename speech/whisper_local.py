"""Local pretrained Whisper (faster-whisper / CTranslate2) on CPU: no API key, no audio leaves the machine.

Browser recordings arrive as WebM/Opus (Chrome, Firefox) or MP4/AAC (Safari). They are decoded with
PyAV (bundled FFmpeg libraries, installed with faster-whisper), resampled to 16 kHz mono, and
transcribed. Silence, undecodable audio and slow transcription come back as explicit statuses, never
as an empty "answer".
"""
from __future__ import annotations

import concurrent.futures
import io
import threading
import time

import numpy as np

from speech.base import Transcription

MIN_SECONDS = 0.3
SILENCE_RMS = 0.003          # below this the recording is treated as silent


def decode_audio(audio: bytes) -> np.ndarray:
    """Any container/codec PyAV understands -> float32 mono 16 kHz."""
    from faster_whisper.audio import decode_audio as fw_decode
    return fw_decode(io.BytesIO(audio), sampling_rate=16000)


class WhisperLocalProvider:
    def __init__(self, model: str = "base.en", timeout_s: float = 30.0, language: str = "en") -> None:
        self.name = f"faster-whisper:{model}"
        self.model_name, self.timeout_s, self.language = model, timeout_s, language
        self._model = None
        self._lock = threading.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _get(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            return self._model

    def _run(self, wav: np.ndarray) -> tuple[str, float | None]:
        segs, _ = self._get().transcribe(wav, language=self.language, beam_size=5, vad_filter=True,
                                         condition_on_previous_text=False)
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        nsp = max((s.no_speech_prob for s in segs), default=None)
        return text, nsp

    def transcribe(self, audio: bytes, content_type: str | None) -> Transcription:
        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000, 1)
        if not audio:
            return Transcription(status="empty", provider=self.name, error="no audio received", latency_ms=ms())
        try:
            wav = decode_audio(audio)
        except Exception as e:
            return Transcription(status="failed", provider=self.name, latency_ms=ms(),
                                 error=f"could not decode the recording ({content_type or 'unknown type'}): "
                                       f"{type(e).__name__}")
        secs = len(wav) / 16000
        if secs < MIN_SECONDS or float(np.sqrt(np.mean(wav ** 2))) < SILENCE_RMS:
            return Transcription(status="empty", provider=self.name, audio_seconds=round(secs, 2), latency_ms=ms(),
                                 error="the recording was silent or too short")
        fut = self._pool.submit(self._run, wav)
        try:
            text, nsp = fut.result(timeout=self.timeout_s)
        except concurrent.futures.TimeoutError:
            return Transcription(status="timeout", provider=self.name, audio_seconds=round(secs, 2), latency_ms=ms(),
                                 error=f"transcription took longer than {self.timeout_s:.0f} s")
        except Exception as e:
            return Transcription(status="failed", provider=self.name, audio_seconds=round(secs, 2), latency_ms=ms(),
                                 error=f"speech model error: {type(e).__name__}: {str(e)[:200]}")
        if not text:
            return Transcription(status="empty", provider=self.name, audio_seconds=round(secs, 2), no_speech_prob=nsp,
                                 latency_ms=ms(), error="no speech recognised")
        return Transcription(status="ok", text=text, provider=self.name, audio_seconds=round(secs, 2),
                             no_speech_prob=None if nsp is None else round(nsp, 3), latency_ms=ms())
