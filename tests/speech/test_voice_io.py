"""Timing behavior for microphone silence detection."""

from __future__ import annotations

import struct
import sys
from types import SimpleNamespace

from openjarvis.speech.voice_io import record_until_silence


class _FakeStream:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = iter(frames)
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def read(self, chunk: int):
        self.reads += 1
        return next(self.frames), False


def _install_audio(monkeypatch, stream: _FakeStream) -> None:
    fake_sd = SimpleNamespace(RawInputStream=lambda **kwargs: stream)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)


def test_initial_silence_stops_at_startup_timeout(monkeypatch) -> None:
    silence = bytes(1024 * 2)
    stream = _FakeStream([silence] * 30)
    _install_audio(monkeypatch, stream)

    record_until_silence(
        sample_rate=1024,
        startup_silence_seconds=2,
        max_seconds=30,
    )

    assert stream.reads == 2


def test_post_speech_uses_normal_silence_window(monkeypatch) -> None:
    speech = struct.pack("1024h", *([1000] * 1024))
    silence = bytes(1024 * 2)
    stream = _FakeStream([speech, silence, silence, silence])
    _install_audio(monkeypatch, stream)

    record_until_silence(
        sample_rate=1024,
        silence_seconds=2,
        startup_silence_seconds=1,
        max_seconds=30,
    )

    assert stream.reads == 3


def test_wait_for_speech_outlasts_startup_window_and_keeps_preroll(
    monkeypatch,
) -> None:
    import io
    import wave

    speech = struct.pack("1024h", *([1000] * 1024))
    silence = bytes(1024 * 2)
    # Far more leading silence than the startup window would allow.
    stream = _FakeStream([silence] * 50 + [speech, silence, silence])
    _install_audio(monkeypatch, stream)

    wav = record_until_silence(
        sample_rate=1024,
        silence_seconds=2,
        startup_silence_seconds=1,
        max_seconds=30,
        wait_for_speech=True,
    )

    assert stream.reads == 53
    with wave.open(io.BytesIO(wav)) as wf:
        # 0.3 s pre-roll at 1 chunk/s rounds to one quiet chunk, then speech
        # and the two-chunk silence tail -- not the 50 chunks of waiting.
        assert wf.getnframes() == 4 * 1024
