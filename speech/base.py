"""Provider interface and result type for speech transcription."""
from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

TranscriptStatus = Literal["ok", "empty", "failed", "timeout", "unavailable"]


class Transcription(BaseModel):
    status: TranscriptStatus
    text: str = ""
    provider: str
    audio_seconds: float | None = None
    no_speech_prob: float | None = None   # provider's own estimate, informational only
    error: str | None = None              # why it failed, shown to the helper
    latency_ms: float | None = None


class SpeechProvider(Protocol):
    name: str

    def transcribe(self, audio: bytes, content_type: str | None) -> Transcription:
        ...


class UnavailableProvider:
    """Used when no speech provider is installed/configured: every request says so explicitly."""

    def __init__(self, reason: str) -> None:
        self.name, self.reason = "unavailable", reason

    def transcribe(self, audio: bytes, content_type: str | None) -> Transcription:
        return Transcription(status="unavailable", provider=self.name, error=self.reason)
